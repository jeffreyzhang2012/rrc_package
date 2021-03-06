import numpy as np
from casadi import *
import time
from datetime import date, datetime
import os

from trifinger_simulation.tasks import move_cube
from rrc_iprl_package.traj_opt.static_object_system import StaticObjectSystem

class StaticObjectOpt:
  def __init__(self,
               nGrid     = 100,
               dt        = 0.1,
               obj_shape = None,
               ):

    self.nGrid = nGrid
    self.dt = dt
    # Define system
    self.system = StaticObjectSystem(
                                     nGrid     = nGrid,
                                     dt        = dt,
                                     obj_shape = obj_shape,
                                    )
    
    qnum = self.system.qnum

    # Get decision variables
    self.t,self.s_flat, self.a = self.system.dec_vars()
    # Pack t,x,u,l into a vector of decision variables
    self.z = self.system.decvar_pack(self.t,self.s_flat, self.a)

    q, dq = self.system.s_unpack(self.s_flat)

    test_q = np.array([[0.5, 0.9, -1.7,0.5, 0.9, -1.7,0.5, 0.9, -1.7,]])
    # Test pinocchio functions with casadi variables
    #print(self.system.get_jacobian(test_q))
    #print(self.system.FK(q[0,:]))

    # Formulate constraints
    self.g, self.lbg, self.ubg = self.get_constraints(self.system, self.t, self.s_flat, self.a)

    # Get cost function
    self.cost = self.cost_func(self.t,self.s_flat,self.a)

    # Concatenate ft_goal and obj_pose params
    self.p =  vertcat(self.system.ft_goal_param, self.system.obj_pose_param)

    # Formulate nlp
    problem = {"x":self.z, "f":self.cost, "g":self.g, "p":self.p}
    options = {"ipopt.print_level":5,
               "ipopt.max_iter":10000,
                "ipopt.tol": 1e-4,
                "print_time": 1
              }
    #options["monitor"] = ["nlp_g"]
    #options = {"monitor":["nlp_f","nlp_g"]}
    self.solver = nlpsol("S", "ipopt", problem, options)

  def solve_nlp(self,
               ft_goal, 
               q0,
               obj_pose  = move_cube.Pose(),
               npz_filepath = None
               ):
                
    qnum = self.system.qnum

    # Get initial guess
    self.z0 = self.system.get_initial_guess(self.z, q0)
    t0, s0, a0 = self.system.decvar_unpack(self.z0)
    self.system.collision_constraint(s0)

    # Path constraints
    self.z_lb, self.z_ub = self.system.path_constraints(self.z, q0, dq0=np.zeros((1,9)), dq_end=np.zeros((1,9)))

    # Set ft_goal and obj_pose param values
    obj_pose_val = np.concatenate((obj_pose.position, obj_pose.orientation))
    p_val = np.concatenate((ft_goal, obj_pose_val))
    
    # Set upper and lower bounds for decision variables
    r = self.solver(x0=self.z0,lbg=self.lbg,ubg=self.ubg,lbx=self.z_lb,ubx=self.z_ub,p=p_val)
    z_soln = r["x"]

    # Final solution and cost
    self.cost = r["f"]
    self.t_soln,self.s_soln,self.a_soln = self.system.decvar_unpack(z_soln)
    self.q_soln, self.dq_soln = self.system.s_unpack(self.s_soln)

    # Get ft positions and velocities at every timestep, in world frame
    # each row is [finger1_x, finger1_y, finger1_z, ..., finger4_x, finger4_y, finger4_z]
    self.ft_pos_soln = np.zeros((self.system.nGrid, self.system.fnum*3))
    self.ft_vel_soln = np.zeros((self.system.nGrid, self.system.fnum*3))
    for t_i in range(self.system.nGrid):
      ft_pos_list = self.system.FK(self.q_soln[t_i, :])

      # Get jacobian
      J = self.system.get_jacobian(self.q_soln[t_i, :])
      # Compute fingertip velocities
      ft_vel = J @ self.dq_soln[t_i, :].T
      self.ft_vel_soln[t_i, :] = ft_vel.T
      for f_i in range(self.system.fnum):
        self.ft_pos_soln[t_i, f_i * qnum: f_i * qnum + qnum] = ft_pos_list[f_i].T

    print("SLACK VARS: {}".format(self.a_soln))

    # Save solver time
    #statistics = self.solver.stats()
    #self.total_time_sec = statistics["t_wall_total"]

    # Save solution
    if npz_filepath is not None:
        np.savez(npz_filepath,
                 dt     = self.system.dt,
                 nGrid  = self.system.nGrid,
                 q0     = q0,
                 ft_goal = ft_goal,
                 obj_pose = obj_pose_val,
                 t      = self.t_soln,
                 q      = self.q_soln,
                 dq     = self.dq_soln,
                 a      = self.a_soln,
                )

  """
  Computes cost
  """
  def cost_func(self,t,s_flat,a):
    cost = 0
    R = np.eye(self.system.fnum * self.system.qnum) * 0.01
    Q = np.eye(self.system.qnum) * 6 # Increase to encourage moving to ft_goal

    q,dq = self.system.s_unpack(s_flat)

    qnum = self.system.qnum

    # Slack variable penalties
    for i in range(a.shape[0]):
      cost += a[i] * 150

    # Add the current distance to fingertip goal
    for i in range(t.shape[0]):
      q_cur = q[i, :]
      ft_cur = self.system.FK(q_cur)
      for f_i in range(self.system.fnum):
        delta = self.system.ft_goal_param[3*f_i:3*f_i+3] - np.squeeze(ft_cur[f_i])
        cost += 0.5 * delta.T @ Q @ delta

    # Add dq to cost, minimize joint velocity..? What is the control input here?
    for i in range(t.shape[0]):
      dq_cur = dq[i, :]
      cost += 0.5 * dq_cur @ R @ dq_cur.T

    # Collision cost
    collision_weight = 0 # Increase to encourage collision avoidance
    #collision_weight = 0.03 # Increase to encourage collision avoidance
    pnorm_min = 1.2
    alpha = 8
    for t_i in range(t.shape[0]):
      q_cur = q[t_i, :]
      for f_i in range(self.system.fnum): # Each finger
        centers = self.system.fingers[f_i].get_sphere_centers_wf(q_cur[qnum*f_i:qnum*f_i+qnum])
        for l_i in [3]:  # Each link
          # radius of spheres on link
          r = self.system.fingers[f_i].r_list[l_i]
          for i in range(centers[l_i].shape[0]): # For each sphere center on link
            c = centers[l_i][i,:]
            pnorm = self.system.get_pnorm_of_pos_wf(c)
            #penalty = fmax(collision_weight * (pnorm_min - pnorm), 0) 
            # smooth max
            penalty = ((pnorm_min - pnorm) * np.e**(alpha*(pnorm_min - pnorm)))/(1 + np.e**(alpha*(pnorm_min - pnorm)))
            cost += penalty * collision_weight

    return cost

  """
  Formulates collocation constraints
  """
  def get_constraints(self,system,t,s,a):

    q,dq = system.s_unpack(s)

    g = [] # constraints
    lbg = [] # constraints lower bound
    ubg = [] # constraints upper bound
  
    # Loop over entire trajectory
    for i in range(t.shape[0] - 1):
      dt = t[i+1] - t[i]
      
      # q and dq trapezoidal integration
      for j in range(system.fnum * system.qnum):
        f = 0.5 * dt * (dq[i+1,j] + dq[i,j]) + q[i,j] - q[i+1,j] 
        g.append(f)
        lbg.append(0)
        ubg.append(0)

    ft_goal_constraints = system.ft_goal_constraint(s, a)
    for r in range(ft_goal_constraints.shape[0]):
      for c in range(ft_goal_constraints.shape[1]):
        g.append(ft_goal_constraints[r,c])
        lbg.append(0)
        ubg.append(np.inf)

    # Collision constraint
    #coll_constraints = system.collision_constraint(s) 
    #for r in range(coll_constraints.shape[0]):
    #  for c in range(coll_constraints.shape[1]):
    #    g.append(coll_constraints[r,c])
    #    lbg.append(0)
    #    ubg.append(np.inf)

    # Fingertip radius constraint
    ft_r_constraints = system.arena_constraint(s) 
    for r in range(ft_r_constraints.shape[0]):
      for c in range(ft_r_constraints.shape[1]):
        g.append(ft_r_constraints[r,c])
        lbg.append(0)
        ubg.append(np.inf)

    return vertcat(*g), vertcat(*lbg), vertcat(*ubg)
    
def main():
  # Get list of desired fingertip positions
  #ft_goal = np.array([0.08457, 0.016751647828266603, 0.07977209510032231,-0.02777764742520991, -0.08161559231227206, 0.07977209510032231,-0.0567923525742952, 0.06486394448412161, 0.07977209510032231])
  ft_goal = np.array([-0.0325, 0, 0,-0.02777764742520991, -0.08161559231227206, 0.07977209510032231,-0.0567923525742952, 0.06486394448412161, 0.07977209510032231])

  q0        = np.array([[0,0.9,-1,0,0.9,-1.7,0,0.9,-1.7]])
  #q0 = np.zeros((1,9))
  #q0[0,1] = 0.7
  
  nGrid = 20
  dt = 0.1

  cube_shape = move_cube._CUBOID_SIZE

  opt_problem = StaticObjectOpt(
               nGrid     = nGrid,
               dt        = dt,
               obj_shape = cube_shape,
               )

  opt_problem.solve_nlp(ft_goal, q0, obj_pose  = move_cube.Pose(position=np.array([0, 0, 0])))

  # Files to save solutions
  today_date = date.today().strftime("%m-%d-%y")
  save_dir = "./logs/{}".format(today_date)
  # Create directory if it does not exist
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_string = "{}/static_object_nGrid_{}_dt_{}".format(save_dir, nGrid, dt) 
  # Save solution in npz file
  np.savez(save_string,
           dt         = opt_problem.dt,
           q0         = q0,
           ft_goal    = ft_goal,
           t          = opt_problem.t_soln,
           q          = opt_problem.q_soln,
           dq         = opt_problem.dq_soln,
           a          = opt_problem.a_soln,
           ft_pos     = opt_problem.ft_pos_soln,
           #obj_shape  = cube_shape,
           fnum       = opt_problem.system.fnum, 
           qnum       = opt_problem.system.qnum, 
          )

if __name__ == "__main__":
  main()

