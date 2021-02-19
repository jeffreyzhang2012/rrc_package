import numpy as np
import enum
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
from casadi import *
from scipy.optimize import nnls

from rrc_iprl_package.control.contact_point import ContactPoint
from trifinger_simulation.tasks import move_cube
from rrc_iprl_package.traj_opt.fixed_contact_point_opt import FixedContactPointOpt
from rrc_iprl_package.traj_opt.fixed_contact_point_system import FixedContactPointSystem
from rrc_iprl_package.traj_opt.static_object_opt import StaticObjectOpt

class PolicyMode(enum.Enum):
        RESET = enum.auto()
        TRAJ_OPT = enum.auto()
        IMPEDANCE = enum.auto()
        RL_PUSH = enum.auto()
        RESIDUAL = enum.auto()

# Object properties
OBJ_MASS = 0.016 # 16 grams
OBJ_SIZE = move_cube._CUBOID_SIZE
OBJ_SIZE_OFFSET = 0.012
OBJ_MU = 1

# Here, hard code the base position of the fingers (as angle on the arena)
r = 0.15
theta_0 = 80
theta_1 = 310
theta_2 = 200
FINGER_BASE_POSITIONS = [
           np.array([[np.cos(theta_0*(np.pi/180))*r, np.sin(theta_0*(np.pi/180))*r, 0]]),
           np.array([[np.cos(theta_1*(np.pi/180))*r, np.sin(theta_1*(np.pi/180))*r, 0]]),
           np.array([[np.cos(theta_2*(np.pi/180))*r, np.sin(theta_2*(np.pi/180))*r, 0]]),
           ]
BASE_ANGLE_DEGREES = [0, -120, -240]

# Information about object faces given face_id
OBJ_FACES_INFO = {
                 1: {"center_param": np.array([0.,-1.,0.]),
                         "face_down_default_quat": np.array([0.707,0,0,0.707]),
                         "adjacent_faces": [6,4,3,5],
                         "opposite_face": 2,
                         "up_axis": np.array([0.,1.,0.]), # UP axis when this face is ground face
                         },
                 2: {"center_param": np.array([0.,1.,0.]),
                         "face_down_default_quat": np.array([-0.707,0,0,0.707]),
                         "adjacent_faces": [6,4,3,5],
                         "opposite_face": 1,
                         "up_axis": np.array([0.,-1.,0.]),
                         },
                 3: {"center_param": np.array([1.,0.,0.]),
                         "face_down_default_quat": np.array([0,0.707,0,0.707]),
                         "adjacent_faces": [1,2,4,6],
                         "opposite_face": 5,
                         "up_axis": np.array([-1.,0.,0.]),
                         },
                 4: {"center_param": np.array([0.,0.,1.]),
                         "face_down_default_quat": np.array([0,1,0,0]),
                         "adjacent_faces": [1,2,3,5],
                         "opposite_face": 6,
                         "up_axis": np.array([0.,0.,-1.]),
                         },
                 5: {"center_param": np.array([-1.,0.,0.]),
                         "face_down_default_quat": np.array([0,-0.707,0,0.707]),
                         "adjacent_faces": [1,2,4,6],
                         "opposite_face": 3,
                         "up_axis": np.array([1.,0.,0.]),
                         },
                 6: {"center_param": np.array([0.,0.,-1.]),
                         "face_down_default_quat": np.array([0,0,0,1]),
                         "adjacent_faces": [1,2,3,5],
                         "opposite_face": 4,
                         "up_axis": np.array([0.,0.,1.]),
                         },
                }

CUBOID_SHORT_FACES = [1,2]
CUBOID_LONG_FACES = [3,4,5,6]

"""
Compute wrench that needs to be applied to object to maintain it on desired trajectory
"""
def track_obj_traj_controller(x_des, dx_des, x_cur, dx_cur, Kp, Kv):
    #print(x_des)
    #print(x_cur.position, x_cur.orientation)
    #print(dx_des)
    #print(dx_cur)
    g = np.array([0, 0, -9.81, 0, 0, 0]) # Gravity vector

    # Force (compute position error)
    p_delta = (x_des[0:3] - x_cur.position)
    dp_delta = (dx_des[0:3] - dx_cur[0:3])
    
    # Moment (compute orientation error)
    # Compute difference between desired and current quaternion
    R_des = Rotation.from_quat(x_des[3:])
    R_cur = Rotation.from_quat(x_cur.orientation)
    o_delta = np.zeros(3)
    for i in range(3):
        o_delta += -0.5 * np.cross(R_cur.as_matrix()[:,i], R_des.as_matrix()[:,i])
    do_delta = (dx_des[3:] - dx_cur[3:]) # is this the angular velocity?

    #print("p_delta: {}".format(p_delta))
    #print("dp_delta: {}".format(dp_delta))
    #print("o_delta: {}".format(o_delta))
    #print("do_delta: {}".format(do_delta))

    # Compute wrench W (6x1) with PD feedback law
    x_delta = np.concatenate((p_delta, -1*o_delta))
    dx_delta = np.concatenate((dp_delta, do_delta))
    W = Kp @ x_delta + Kv @ dx_delta - OBJ_MASS * g
    
    print("x_delta: {}".format(x_delta))
    print("dx_delta: {}".format(dx_delta))

    #print(W)

    return W

"""
Compute fingertip forces necessary to keep object on desired trajectory
"""
def get_ft_forces(x_des, dx_des, x_cur, dx_cur, Kp, Kv, cp_params):
    # Get desired wrench for object COM to track obj traj
    W = track_obj_traj_controller(x_des, dx_des, x_cur, dx_cur, Kp, Kv)

    # Get list of contact point positions and orientations in object frame
    # By converting cp_params to contactPoints
    cp_list = []
    for cp_param in cp_params:
        if cp_param is not None:
            cp = get_cp_of_from_cp_param(cp_param)
            cp_list.append(cp)
    fnum = len(cp_list)

    # To compute grasp matrix
    G = __get_grasp_matrix(np.concatenate((x_cur.position, x_cur.orientation)), cp_list)
    
    # Solve for fingertip forces via optimization
    # TODO use casadi for now, make new problem every time. If too slow, try cvxopt or scipy.minimize, 
    # Or make a parametrized problem??
    # Contact-surface normal vector for each contact point
    n = np.array([1, 0, 0]) # contact point frame x axis points into object    
    # Tangent vectors d_i for each contact point
    d = [np.array([0, 1, 0]),
         np.array([0, -1, 0]),
         np.array([0, 0, 1]),
         np.array([0, 0, -1])]
    V = np.zeros((9,12));
    for i in range(3):
        for j in range(4):
            V[i*3:(i+1)*3,i*4+j] = n + OBJ_MU * d[j]
    B_soln = nnls(G@V,W)[0]
    L = V@B_soln

    # Formulate optimization problem
    #B = SX.sym("B", len(d) * fnum) # Scaling weights for each of the cone vectors
    #B0 = np.zeros(B.shape[0]) # Initial guess for weights

    ## Fill lambda vector
    #l_list = []
    #for j in range(fnum):
    #    l = 0 # contact force
    #    for i in range(len(d)):
    #        v = n + OBJ_MU * d[i] 
    #        l += B[j*fnum + i] * v 
    #    l_list.append(l)
    #L = vertcat(*l_list) # (9x1) lambda vector

    #f = G @ L - W # == 0

    ## Formulate constraints
    #g = f # contraint function
    #g_lb = np.zeros(f.shape[0]) # constraint lower bound
    #g_ub = np.zeros(f.shape[0]) # constraint upper bound

    ## Constraints on B
    #z_lb = np.zeros(B.shape[0]) # Lower bound on beta
    #z_ub = np.ones(B.shape[0]) * np.inf # Upper bound on beta

    #cost = L.T @ L

    #problem = {"x": B, "f": cost, "g": g}
    #options = {"ipopt.print_level":5,
    #           "ipopt.max_iter":10000,
    #            "ipopt.tol": 1e-4,
    #            "print_time": 1
    #          }
    #solver = nlpsol("S", "ipopt", problem, options)
    #r = solver(x0=B0, lbg=g_lb, ubg=g_ub, lbx=z_lb, ubx=z_ub)
    #B_soln = r["x"]

    # Compute contact forces in contact point frames from B_soln
    # TODO fix list length when there are only 2 contact points
    # save for later since we always have a 3 fingered grasp
    l_wf_soln = []
    for j in range(fnum):
        l_cf = 0 # contact force
        for i in range(len(d)):
            v = n + OBJ_MU * d[i] 
            l_cf += B_soln[j*fnum + i] * v 

        # Convert from contact point frame to world frame
        cp = cp_list[j]
        R_cp_2_o = Rotation.from_quat(cp.quat_of)
        R_o_2_w = Rotation.from_quat(x_cur.orientation)
        l_wf = R_o_2_w.apply(R_cp_2_o.apply(np.squeeze(l_cf)))
        l_wf_soln.append(l_wf)
    return l_wf_soln, W


"""
Compute joint torques to move fingertips to desired locations
Inputs:
tip_pos_desired_list: List of desired fingertip positions for each finger
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""
def impedance_controller(
                       tip_pos_desired_list,
                       tip_vel_desired_list,
                       q_current,  
                       dq_current,
                       custom_pinocchio_utils,
                       tip_forces_wf = None,
                       Kp                   = [25,25,25,25,25,25,25,25,25],
                       Kv                   = [1,1,1,1,1,1,1,1,1],
                       ):
    torque = 0
    for finger_id in range(3):
        # Get contact forces for single finger
        if tip_forces_wf is None:
            f_wf = None
        else:
            f_wf = np.expand_dims(np.array(tip_forces_wf[finger_id * 3:finger_id*3 + 3]),1)

        finger_torque = impedance_controller_single_finger(
                                                    finger_id,
                                                    tip_pos_desired_list[finger_id],
                                                    tip_vel_desired_list[finger_id],
                                                    q_current,
                                                    dq_current,
                                                    custom_pinocchio_utils,
                                                    tip_force_wf = f_wf,
                                                    Kp                   = Kp,
                                                    Kv                   = Kv,
                                                    )
        torque += finger_torque
    return torque

"""
Compute joint torques to move fingertip to desired location
Inputs:
finger_id: Finger 0, 1, or 2
tip_desired: Desired fingertip pose **ORIENTATION??**
    for orientation: transform fingertip reference frame to world frame (take into account object orientation)
    for now, just track position
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""
def impedance_controller_single_finger(
    finger_id,
    tip_pos_desired,
    tip_vel_desired,
    q_current,
    dq_current,
    custom_pinocchio_utils,
    tip_force_wf = None,
    Kp           = [25,25,25,25,25,25,25,25,25],
    Kv           = [1,1,1,1,1,1,1,1,1]
    ):
    
    Kp_x = Kp[finger_id*3 + 0]
    Kp_y = Kp[finger_id*3 + 1]
    Kp_z = Kp[finger_id*3 + 2]
    Kp = np.diag([Kp_x, Kp_y, Kp_z])

    Kv_x = Kv[finger_id*3 + 0]
    Kv_y = Kv[finger_id*3 + 1]
    Kv_z = Kv[finger_id*3 + 2]
    Kv = np.diag([Kv_x, Kv_y, Kv_z])

    # Compute current fingertip position
    x_current = custom_pinocchio_utils.forward_kinematics(q_current)[finger_id]

    delta_x = np.expand_dims(np.array(tip_pos_desired) - np.array(x_current), 1)
    #print("Current x: {}".format(x_current))
    #print("Desired x: {}".format(tip_desired))
    #print("Delta: {}".format(delta_x))
    
    # Get full Jacobian for finger
    Ji = custom_pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
    # Just take first 3 rows, which correspond to linear velocities of fingertip
    Ji = Ji[:3, :]

    # Get g matrix for gravity compensation
    _, g = custom_pinocchio_utils.get_lambda_and_g_matrix(finger_id, q_current, Ji)

    # Get current fingertip velocity
    dx_current = Ji @ np.expand_dims(np.array(dq_current), 1)

    delta_dx = np.expand_dims(np.array(tip_vel_desired),1) - np.array(dx_current)

    if tip_force_wf is not None:
        torque = np.squeeze(Ji.T @ (Kp @ delta_x + Kv @ delta_dx) + Ji.T @ tip_force_wf) + g
    else:
        torque = np.squeeze(Ji.T @ (Kp @ delta_x + Kv @ delta_dx)) + g

    #print("Finger {} delta".format(finger_id))
    #print(np.linalg.norm(delta_x))
    return torque

"""
Compute contact point position in world frame
Inputs:
cp_param: Contact point param [px, py, pz]
cube: Block object, which contains object shape info
"""
def get_cp_pos_wf_from_cp_param(cp_param, cube_pos_wf, cube_quat_wf, use_obj_size_offset = False):
    cp = get_cp_of_from_cp_param(cp_param, use_obj_size_offset = use_obj_size_offset)

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(cp.pos_of) + translation

"""
Get contact point positions in world frame from cp_params
"""
def get_cp_pos_wf_from_cp_params(cp_params, cube_pos, cube_quat, use_obj_size_offset = False):
    # Get contact points in wf
    fingertip_goal_list = []
    for i in range(len(cp_params)):
        if cp_params[i] is None:
            fingertip_goal_list.append(None)
        else:
            fingertip_goal_list.append(get_cp_pos_wf_from_cp_param(cp_params[i], cube_pos, cube_quat, use_obj_size_offset = use_obj_size_offset))
    return fingertip_goal_list

"""
Compute contact point position in object frame
Inputs:
cp_param: Contact point param [px, py, pz]
"""
def get_cp_of_from_cp_param(cp_param, use_obj_size_offset = False):
    cp_of = []
    # Get cp position in OF
    for i in range(3):
        if use_obj_size_offset:
            cp_of.append(-(OBJ_SIZE[i] + OBJ_SIZE_OFFSET)/2 + (cp_param[i]+1)*(OBJ_SIZE[i] + OBJ_SIZE_OFFSET)/2)
        else:
            cp_of.append(-OBJ_SIZE[i]/2 + (cp_param[i]+1)*OBJ_SIZE[i]/2)

    cp_of = np.asarray(cp_of)

    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        quat = (np.sqrt(2)/2, 0, 0, np.sqrt(2)/2)
    elif y_param == 1:
        quat = (np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2)
    elif x_param == 1:
        quat = (0, 0, 1, 0)
    elif z_param == 1:
        quat = (np.sqrt(2)/2, 0, np.sqrt(2)/2, 0)
    elif x_param == -1:
        quat = (1, 0, 0, 0)
    elif z_param == -1:
        quat = (np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0)

    cp = ContactPoint(cp_of, quat)
    return cp

"""
Get face id on cube, given cp_param
cp_param: [x,y,z]
"""
def get_face_from_cp_param(cp_param):
    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        face = 1
    elif y_param == 1:
        face = 2
    elif x_param == 1:
        face = 3
    elif z_param == 1:
        face = 4
    elif x_param == -1:
        face = 5
    elif z_param == -1:
        face = 6

    return face

"""
Trasform point p from world frame to object frame, given object pose
"""
def get_wf_from_of(p, obj_pose):
    cube_pos_wf = obj_pose.position
    cube_quat_wf = obj_pose.orientation

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)
    
    return rotation.apply(p) + translation

"""
Trasform point p from object frame to world frame, given object pose
"""
def get_of_from_wf(p, obj_pose):
    cube_pos_wf = obj_pose.position
    cube_quat_wf = obj_pose.orientation

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)
    
    rotation_inv = rotation.inv()
    translation_inv = -rotation_inv.apply(translation)

    return rotation_inv.apply(p) + translation_inv

##############################################################################
# Lift mode functions
##############################################################################
"""
Run trajectory optimization
obj_pose: current object pose (for getting contact points)
current_position: current joint positions of robot
x0: object initial position for traj opt
x_goal: object goal position for traj opt
nGrid: number of grid points
dt: delta t
"""
def run_fixed_cp_traj_opt(obj_pose, cp_params, current_position, custom_pinocchio_utils, x0, x_goal, nGrid, dt, npz_filepath = None):

    cp_params_on_obj = []
    for cp in cp_params:
        if cp is not None: cp_params_on_obj.append(cp)
    fnum = len(cp_params_on_obj)

    # Formulate and solve optimization problem
    opt_problem = FixedContactPointOpt(
                                      nGrid     = nGrid,    # Number of timesteps
                                      dt        = dt,       # Length of each timestep (seconds)
                                      fnum      = fnum,
                                      cp_params = cp_params_on_obj,
                                      x0        = x0,
                                      x_goal    = x_goal,
                                      obj_shape = OBJ_SIZE,
                                      obj_mass  = OBJ_MASS,
                                      npz_filepath = npz_filepath
                                      )
    
    x_soln     = np.array(opt_problem.x_soln)
    dx_soln    = np.array(opt_problem.dx_soln)
    l_wf_soln  = np.array(opt_problem.l_wf_soln)

    return x_soln, dx_soln, l_wf_soln
    

"""
Get initial contact points on cube
Assign closest cube face to each finger
Since we are lifting object, don't worry about wf z-axis, just care about wf xy-plane
"""
def get_lifting_cp_params(obj_pose):
    # TODO this is assuming that cuboid is always resting on one of its long sides
    # face that is touching the ground
    ground_face = get_closest_ground_face(obj_pose)

    # Transform finger base positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, obj_pose)
        finger_base_of.append(f_of)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1,0])
    y_axis = np.array([0,1])
    
    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)
        
    xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
        
        xy_distances[f_i, 0] = np.sign(f_of[0,y_ind]) * x_dist
        xy_distances[f_i, 1] = np.sign(f_of[0,x_ind]) * y_dist
    
    free_faces = \
        [x for x in OBJ_FACES_INFO[ground_face]["adjacent_faces"] if x not in CUBOID_SHORT_FACES]

    # For each face, choose closest finger
    finger_assignments = {}
    for face in free_faces:
        face_ind = OBJ_FACES_INFO[ground_face]["adjacent_faces"].index(face)
        if face_ind in [2,3]:
            # Check y_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][x_ind] < 0: 
                # Want most negative value
                f_i = np.nanargmin(xy_distances[:,1])
            else:
                # Want most positive value
                f_i = np.nanargmax(xy_distances[:,1])
        else:
            # Check x_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][y_ind] < 0: 
                f_i = np.nanargmin(xy_distances[:,0])
            else:
                f_i = np.nanargmax(xy_distances[:,0])
        finger_assignments[face] = [f_i]
        xy_distances[f_i, :] = np.nan

    # Assign last finger to one of the long faces
    max_ind = np.unravel_index(np.nanargmax(xy_distances), xy_distances.shape)
    curr_finger_id = max_ind[0] 
    face = assign_faces_to_fingers(obj_pose, [curr_finger_id], free_faces)[curr_finger_id]
    finger_assignments[face].append(curr_finger_id)

    print("finger assignments: {}".format(finger_assignments))
    
    # Set contact point params for two long faces
    cp_params = [None, None, None]
    height_param = -0.85 # Always want cps to be at this height
    width_param = 0.5 # Always want cps to be at this height

    for face, finger_id_list in finger_assignments.items():
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        param += OBJ_FACES_INFO[OBJ_FACES_INFO[ground_face]["opposite_face"]]["center_param"] * height_param
        if len(finger_id_list) == 2:
            # Find the closest short face to each finger
            nearest_short_faces = assign_faces_to_fingers(obj_pose,
                                                          finger_id_list,
                                                          CUBOID_SHORT_FACES.copy())
    
            for f_i, short_face in nearest_short_faces.items():
                new_param = param.copy()
                new_param += OBJ_FACES_INFO[short_face]["center_param"] * width_param
                cp_params[f_i] = new_param
                
        else:
            cp_params[finger_id_list[0]] = param
    print("LIFT CP PARAMS: {}".format(cp_params))

    return cp_params

"""
For a specified finger f_i and list of available faces, get closest face
"""
def assign_faces_to_fingers(obj_pose, finger_id_list, free_faces):

    ground_face = get_closest_ground_face(obj_pose)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1,0])
    y_axis = np.array([0,1])

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)

    # Transform finger base positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, obj_pose)
        finger_base_of.append(f_of)

    xy_distances = np.zeros((3, 2)) # Rows: fingers, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
        
        xy_distances[f_i, 0] = x_dist
        xy_distances[f_i, 1] = y_dist

    assignments = {}
    for i in range(3):
        max_ind = np.unravel_index(np.nanargmax(xy_distances), xy_distances.shape)
        f_i = max_ind[0]
        if f_i not in finger_id_list:
            xy_distances[f_i, :] = np.nan
            continue
        furthest_axis = max_ind[1]
        x_dist = xy_distances[f_i, 0]
        y_dist = xy_distances[f_i, 1]
        if furthest_axis == 0: # distance to x axis is greater than to y axis
            if finger_base_of[f_i][0, y_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1] # 2
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0] # 1
        else:
            if finger_base_of[f_i][0, x_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2] # 3
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3] # 5

        # Get alternate closest face
        if face not in free_faces:
            alternate_axis = abs(furthest_axis - 1)
            if alternate_axis == 0:
                if finger_base_of[f_i][0, y_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1] # 2
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0] # 1
            else:
                if finger_base_of[f_i][0, x_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2] # 3
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3] # 5

        assignments[f_i] = face

        xy_distances[f_i, :] = np.nan
        free_faces.remove(face)

    return assignments

def get_pre_grasp_ft_goal(obj_pose, fingertips_current_wf, cp_params):
    ft_goal = np.zeros(9)
    incr = 0.03

    # Get list of desired fingertip positions
    cp_wf_list = get_cp_pos_wf_from_cp_params(cp_params, obj_pose.position, obj_pose.orientation, use_obj_size_offset = True)

    for f_i in range(3):
        f_wf = cp_wf_list[f_i]
        if cp_params[f_i] is None:
            f_new_wf = fingertips_current_wf[f_i]
        else:
            # Get face that finger is on
            face = get_face_from_cp_param(cp_params[f_i])
            f_of = get_of_from_wf(f_wf, obj_pose)

            # Release object
            f_new_of = f_of - incr * OBJ_FACES_INFO[face]["up_axis"]

            # Convert back to wf
            f_new_wf = get_wf_from_of(f_new_of, obj_pose)

        ft_goal[3*f_i:3*f_i+3] = f_new_wf
    return ft_goal

"""
Set up traj opt for fingers and static object
"""
def define_static_object_opt(nGrid, dt):
    problem = StaticObjectOpt(
                 nGrid     = nGrid,
                 dt        = dt,
                 obj_shape = OBJ_SIZE,
                 )
    return problem

"""
Solve traj opt to get finger waypoints
"""
def get_finger_waypoints(nlp, ft_goal, q_cur, obj_pose, npz_filepath = None):
    nlp.solve_nlp(ft_goal, q_cur, obj_pose = obj_pose, npz_filepath = npz_filepath)
    ft_pos = nlp.ft_pos_soln
    ft_vel = nlp.ft_vel_soln
    return ft_pos, ft_vel

##############################################################################
# Flip mode functions
##############################################################################

"""
Determine face that is closest to ground
"""
def get_closest_ground_face(obj_pose):
    min_z = np.inf
    min_face = None
    for i in range(1,7):
        c = OBJ_FACES_INFO[i]["center_param"].copy()
        c_wf = get_wf_from_of(c, obj_pose)
        if c_wf[2] < min_z:
            min_z = c_wf[2]
            min_face = i

    return min_face

"""
Get flipping contact points
"""
def get_flipping_cp_params(
                           init_pose,
                           goal_pose,
                          ):
    # Get goal face
    init_face = get_closest_ground_face(init_pose)
    #print("Init face: {}".format(init_face))
    # Get goal face
    goal_face = get_closest_ground_face(goal_pose)
    #print("Goal face: {}".format(goal_face))
    
    if goal_face not in OBJ_FACES_INFO[init_face]["adjacent_faces"]:
        #print("Goal face not adjacent to initial face")
        goal_face = OBJ_FACES_INFO[init_face]["adjacent_faces"][0]
        #print("Intermmediate goal face: {}".format(goal_face))

    # Common adjacent faces to init_face and goal_face
    common_adjacent_faces = list(set(OBJ_FACES_INFO[init_face]["adjacent_faces"]). intersection(OBJ_FACES_INFO[goal_face]["adjacent_faces"]))

    opposite_goal_face = OBJ_FACES_INFO[goal_face]["opposite_face"]
    
    #print("place fingers on faces {}, towards face {}".format(common_adjacent_faces, opposite_goal_face))

    # Find closest fingers to each of the common_adjacent_faces
    # Transform finger tip positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, init_pose)
        #f_of = np.squeeze(get_of_from_wf(f_wf, init_pose))
        finger_base_of.append(f_of)

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(init_face)
    # Find distance from x axis and y axis, and store in xy_distances
    x_axis = np.array([1,0])
    y_axis = np.array([0,1])
        
    xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
        
        xy_distances[f_i, 0] = np.sign(f_of[0,y_ind]) * x_dist
        xy_distances[f_i, 1] = np.sign(f_of[0,x_ind]) * y_dist

    finger_assignments = {}
    for face in common_adjacent_faces:
        face_ind = OBJ_FACES_INFO[init_face]["adjacent_faces"].index(face)
        if face_ind in [2,3]:
            # Check y_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][x_ind] < 0: 
                # Want most negative value
                f_i = np.nanargmin(xy_distances[:,1])
            else:
                # Want most positive value
                f_i = np.nanargmax(xy_distances[:,1])
        else:
            # Check x_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][y_ind] < 0: 
                f_i = np.nanargmin(xy_distances[:,0])
            else:
                f_i = np.nanargmax(xy_distances[:,0])
        finger_assignments[face] = f_i
        xy_distances[f_i, :] = np.nan

    cp_params = [None, None, None]
    # TODO Hardcoded
    height_param = -0.65 # Always want cps to be at this height
    width_param = 0.65 
    for face in common_adjacent_faces:
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        param += OBJ_FACES_INFO[OBJ_FACES_INFO[init_face]["opposite_face"]]["center_param"] * height_param
        param += OBJ_FACES_INFO[opposite_goal_face]["center_param"] * width_param
        cp_params[finger_assignments[face]] = param
        #cp_params.append(param)
    #print("Assignments: {}".format(finger_assignments))
    return cp_params, init_face, goal_face

##############################################################################
# Private functions
##############################################################################

"""
Given a ground face id, get the axes that are parallel to the floor
"""
def __get_parallel_ground_plane_xy(ground_face):
    if ground_face in [1,2]:
        x_ind = 0
        y_ind = 2
    if ground_face in [3,5]:
        x_ind = 2
        y_ind = 1
    if ground_face in [4,6]:
        x_ind = 0
        y_ind = 1
    return x_ind, y_ind

"""
Get distance from point to line (in 2D)
Inputs:
a, b: points on line
p: standalone point, for which we want to compute its distance to line
"""
def __get_distance_from_pt_2_line(a, b, p):
    a = np.squeeze(a)
    b = np.squeeze(b)
    p = np.squeeze(p)

    ba = b - a
    ap = a - p
    c = ba * (np.dot(ap,ba) / np.dot(ba,ba))
    d = ap - c
    
    return np.sqrt(np.dot(d,d))

"""
Get grasp matrix
Input:
x: object pose [px, py, pz, qw, qx, qy, qz]
"""
def __get_grasp_matrix(x, cp_list):
    fnum = len(cp_list)
    obj_dof = 6

    # Contact model force selection matrix
    l_i = 3
    H_i = np.array([
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  ])
    H = np.zeros((l_i*fnum,obj_dof*fnum))
    for i in range(fnum):
      H[i*l_i:i*l_i+l_i, i*obj_dof:i*obj_dof+obj_dof] = H_i
    
    # Transformation matrix from object frame to world frame
    quat_o_2_w = [x[3], x[4], x[5], x[6]]

    G_list = []

    # Calculate G_i (grasp matrix for each finger)
    for c in cp_list:
        cp_pos_of = c.pos_of # Position of contact point in object frame
        quat_cp_2_o = c.quat_of # Orientation of contact point frame w.r.t. object frame

        S = np.array([
                     [0, -cp_pos_of[2], cp_pos_of[1]],
                     [cp_pos_of[2], 0, -cp_pos_of[0]],
                     [-cp_pos_of[1], cp_pos_of[0], 0]
                     ])

        P_i = np.eye(6)
        P_i[3:6,0:3] = S

        # Orientation of cp frame w.r.t. world frame
        # quat_cp_2_w = quat_o_2_w * quat_cp_2_o
        R_cp_2_w = Rotation.from_quat(quat_o_2_w) * Rotation.from_quat(quat_cp_2_o)
        # R_i is rotation matrix from contact frame i to world frame
        R_i = R_cp_2_w.as_matrix()
        R_i_bar = np.zeros((6,6))
        R_i_bar[0:3,0:3] = R_i
        R_i_bar[3:6,3:6] = R_i

        G_iT = R_i_bar.T @ P_i.T
        G_list.append(G_iT)
    
    GT_full = np.concatenate(G_list)
    GT = H @ GT_full
    #print(GT.T)
    return GT.T

"""
Get matrix to convert dquat (4x1 vector) to angular velocities (3x1 vector)
"""
def get_dquat_to_dtheta_matrix(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    M = np.array([
                  [-qx, -qy, -qz],
                  [qw, qz, -qy],
                  [-qz, qw, qx],
                  [qy, -qx, qw],
                ])

    return M.T

def get_ft_R(q):
    R_list = []
    for f_i, angle in enumerate(BASE_ANGLE_DEGREES):
        theta = angle * (np.pi/180)
        q1 = q[3*f_i + 0]
        q2 = q[3*f_i + 1]
        q3 = q[3*f_i + 2]
        R = np.array([[np.cos(q1)*np.cos(theta), (np.sin(q1)*np.sin(q2)*np.cos(theta) - np.sin(theta)*np.cos(q2))*np.cos(q3) + (np.sin(q1)*np.cos(q2)*np.cos(theta) + np.sin(q2)*np.sin(theta))*np.sin(q3), -(np.sin(q1)*np.sin(q2)*np.cos(theta) - np.sin(theta)*np.cos(q2))*np.sin(q3) + (np.sin(q1)*np.cos(q2)*np.cos(theta) + np.sin(q2)*np.sin(theta))*np.cos(q3)], [np.sin(theta)*np.cos(q1), (np.sin(q1)*np.sin(q2)*np.sin(theta) + np.cos(q2)*np.cos(theta))*np.cos(q3) + (np.sin(q1)*np.sin(theta)*np.cos(q2) - np.sin(q2)*np.cos(theta))*np.sin(q3), -(np.sin(q1)*np.sin(q2)*np.sin(theta) + np.cos(q2)*np.cos(theta))*np.sin(q3) + (np.sin(q1)*np.sin(theta)*np.cos(q2) - np.sin(q2)*np.cos(theta))*np.cos(q3)], [-np.sin(q1), np.sin(q2)*np.cos(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2), -np.sin(q2)*np.sin(q3)*np.cos(q1) + np.cos(q1)*np.cos(q2)*np.cos(q3)]])
        R_list.append(R)
    return R_list
