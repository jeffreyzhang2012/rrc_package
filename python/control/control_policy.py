"""
Implements ImpedanceControllerPolicy class which returns actions to be compatible
with Gym environment
"""

import os
import os.path as osp
import numpy as np
import joblib
import enum
import copy
import time
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import csv

import datetime
from trifinger_simulation import TriFingerPlatform
from trifinger_simulation.tasks import move_cube
from trifinger_simulation import pinocchio_utils
from trifinger_simulation import visual_objects

from rrc_iprl_package.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_iprl_package.control import controller_utils as c_utils
from rrc_iprl_package.control.controller_utils import PolicyMode

try:
    import torch
except ImportError:
    torch = None


class TrajMode(enum.Enum):
    RESET = enum.auto()
    PRE_TRAJ_LOWER = enum.auto()
    PRE_TRAJ_REACH = enum.auto()
    REPOSE = enum.auto()


class ImpedanceControllerPolicy:
    USE_FILTERED_POSE = True

    KP = [300, 300, 400,
          300, 300, 400,
          300, 300, 400]
    KV = [0.7, 0.7, 0.8,
          0.7, 0.7, 0.8,
          0.7, 0.7, 0.8]

    #KP_REPOSE = [130, 130, 130,
    #             130, 130, 130,
    #             130, 130, 130]
    #KV_REPOSE = [0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7,
    #             0.7, 0.7, 0.7]

    KP_REPOSE = KP
    KV_REPOSE = KV

    kp_obj = 0.01
    KP_OBJ = [kp_obj,
              kp_obj,
              kp_obj,
              kp_obj,
              kp_obj,
              kp_obj,]

    kv_obj = 0.003
    KV_OBJ = [kv_obj,
              kv_obj,
              kv_obj,
              kv_obj,
              kv_obj,
              kv_obj,]

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, debug_waypoints=False, difficulty=None):
        self.difficulty = difficulty
        self.action_space = action_space
        self.debug_waypoints = debug_waypoints
        self.set_init_goal(initial_pose, goal_pose)
        self.init_face = None
        self.goal_face = None
        self.platform = None
        print("USE_FILTERED_POSE: {}".format(self.USE_FILTERED_POSE))
        print("KP: {}".format(self.KP))
        print("KV: {}".format(self.KV))
        print("KP_REPOSE: {}".format(self.KP_REPOSE))
        print("KV_REPOSE: {}".format(self.KV_REPOSE))
        print("KP_OBJ: {}".format(self.KP_OBJ))
        print("KV_OBJ: {}".format(self.KV_OBJ))

        self.initialize_logging()

    def initialize_logging(self):
        # CSV logging file path # need leading / for singularity image
        if osp.exists("/output"):
            self.csv_filepath           = "/output/control_policy_data.csv"
            self.grasp_trajopt_filepath = "/output/grasp_trajopt_data"
            self.lift_trajopt_filepath  = "/output/lift_trajopt_data"
            self.control_policy_log_filepath = "/output/control_policy_log"
        else:
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if not osp.exists("./output/{}".format(time_str)):
                os.makedirs("./output/{}".format(time_str))
            self.csv_filepath           = "./output/{}/control_policy_data.csv".format(time_str)
            self.grasp_trajopt_filepath = "./output/{}/grasp_trajopt_data".format(time_str)
            self.lift_trajopt_filepath  = "./output/{}/lift_trajopt_data".format(time_str)
            self.control_policy_log_filepath = "./output/{}/control_policy_log".format(time_str)

        # Lists for logging data
        # Rows correspond to step_count / timestamp
        self.l_step_count       = [] # step_count
        self.l_timestamp        = [] # time
        self.l_desired_ft_pos   = [] # fingertip positions - desired
        self.l_actual_ft_pos    = [] # fingertip positions - actual (computed from observation)
        self.l_desired_ft_vel   = [] # fingertip velocities - desired
        self.l_desired_obj_pose = [] # object position - desired
        self.l_desired_obj_vel = [] # object position - desired
        self.l_observed_obj_pose = [] # object position - observed
        self.l_observed_filt_obj_pose = [] # object position - observed
        self.l_observed_obj_vel = [] # object velocity - observed
        self.l_desired_ft_force = [] # fingerip forces - desired
        self.l_desired_torque = []

        # Logs for debugging object pose feedback controller
        self.DEBUG = True
        self.l_dquat = [] # quaternion derivatives
        self.l_desired_obj_w = []

    """
    Store logs in npz file
    """
    def save_log(self):
        np.savez(self.control_policy_log_filepath,
                 step_count       = np.asarray(self.l_step_count),
                 timestamp        = np.asarray(self.l_timestamp),
                 desired_ft_pos   = np.asarray(self.l_desired_ft_pos),
                 desired_ft_vel   = np.asarray(self.l_desired_ft_vel),
                 actual_ft_pos    = np.asarray(self.l_actual_ft_pos),
                 desired_obj_pose = np.squeeze(np.asarray(self.l_desired_obj_pose)),
                 desired_obj_vel = np.squeeze(np.asarray(self.l_desired_obj_vel)),
                 observed_obj_pose = np.squeeze(np.asarray(self.l_observed_obj_pose)),
                 observed_filt_obj_pose = np.squeeze(np.asarray(self.l_observed_filt_obj_pose)),
                 observed_obj_vel = np.squeeze(np.asarray(self.l_observed_obj_vel)),
                 desired_ft_force = np.squeeze(np.asarray(self.l_desired_ft_force)),
                 desired_torque   = np.squeeze(np.asarray(self.l_desired_torque)),
                 dquat            = np.squeeze(np.asarray(self.l_dquat)),
                 desired_obj_w    = np.squeeze(np.asarray(self.l_desired_obj_w)),
                )

    def reset_policy(self, observation, platform=None):
        if platform:
            self.platform = platform
        self.custom_pinocchio_utils = CustomPinocchioUtils(
                self.platform.simfinger.finger_urdf_path,
                self.platform.simfinger.tip_link_names)

        # Define nlp for finger traj opt
        nGrid = 40
        dt = 0.04
        self.finger_nlp = c_utils.define_static_object_opt(nGrid, dt)

        init_position = np.array([0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])
        self.init_ft_pos = self.get_fingertip_pos_wf(init_position)
        self.init_ft_pos = np.asarray(self.init_ft_pos).flatten()

        # Previous object pose and time (for estimating object velocity)
        self.prev_obj_pose = get_pose_from_observation(observation)
        if osp.exists("/output"):
            self.prev_step_time = observation["cam0_timestamp"]
        else:
            self.prev_step_time = observation["cam0_timestamp"] / 1000
        self.prev_vel = np.zeros(6)
        self.filt_vel = np.zeros(6)

        self.filtered_obj_pose = get_pose_from_observation(observation)

        self.ft_pos_traj = np.tile(self.init_ft_pos, (10000,1))
        self.ft_vel_traj = np.zeros((10000,9))
        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None
        self.mode = TrajMode.RESET
        self.plan_trajectory(observation)

        # Counters
        self.step_count = 0 # Number of times predict() is called

    def set_init_goal(self, initial_pose, goal_pose, flip=False):
        self.goal_pose = goal_pose
        self.x0 = np.concatenate([initial_pose.position, initial_pose.orientation])[None]
        init_goal_dist = np.linalg.norm(goal_pose.position - initial_pose.position)
        #print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}, '
        #      f'dist: {init_goal_dist}')
        #print(f'init orientation: {initial_pose.orientation}, goal orientation: {goal_pose.orientation}')

    """
    Get contact point parameters for either lifting
    """
    def set_cp_params(self, observation):
        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        self.cp_params = c_utils.get_lifting_cp_params(obj_pose)

    """
    Run trajectory optimization to move object given fixed contact points
    """
    def set_traj_lift_object(self, observation, nGrid = 50, dt = 0.01):
        self.traj_waypoint_counter = 0
        qnum = 3

        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        # Clip obj z coord to half width of cube
        clipped_pos = obj_pose.position.copy()
        clipped_pos[2] = 0.01
        #clipped_pos[2] = max(obj_pose.position[2], move_cube._CUBOID_SIZE[0]/2)
        x0 = np.concatenate([clipped_pos, obj_pose.orientation])[None]
        x_goal = x0.copy()
        x_goal[0, :3] = self.goal_pose.position
        if self.difficulty == 4:
            x_goal[0, -4:] = self.goal_pose.orientation

        print("Object pose position: {}".format(obj_pose.position))
        print("Object pose orientation: {}".format(obj_pose.orientation))
        print("Traj lift x0: {}".format(repr(x0)))
        print("Traj lift x_goal: {}".format(repr(x_goal)))

        # Get current joint positions
        current_position, _ = get_robot_position_velocity(observation)
        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        self.x_soln, self.dx_soln, l_wf_soln = c_utils.run_fixed_cp_traj_opt(
                obj_pose, self.cp_params, current_position, self.custom_pinocchio_utils,
                x0, x_goal, nGrid, dt, npz_filepath = self.lift_trajopt_filepath)

        ft_pos = np.zeros((nGrid, 9))
        ft_vel = np.zeros((nGrid, 9))

        free_finger_id = None
        for i, cp in enumerate(self.cp_params):
            if cp is None:
                free_finger_id = i
                break

        for t_i in range(nGrid):
            # Set fingertip goal positions and velocities from x_soln, dx_soln
            next_cube_pos_wf = self.x_soln[t_i, 0:3]
            next_cube_quat_wf = self.x_soln[t_i, 3:]

            ft_pos_list = c_utils.get_cp_pos_wf_from_cp_params(
                    self.cp_params, next_cube_pos_wf, next_cube_quat_wf)

            # Hold free_finger at current ft position
            if free_finger_id is not None:
                ft_pos_list[free_finger_id] = current_ft_pos[free_finger_id]
            ft_pos[t_i, :] = np.asarray(ft_pos_list).flatten()

            # Fingertip velocities
            ft_vel_arr = np.tile(self.dx_soln[t_i, 0:3], 3)
            if free_finger_id is not None:
                ft_vel_arr[free_finger_id * qnum : free_finger_id * qnum + qnum] = np.zeros(qnum)
            ft_vel[t_i, :] = ft_vel_arr

        # Add 0 forces for free_fingertip to l_wf
        l_wf = np.zeros((nGrid, 9))
        i = 0
        for f_i in range(3):
            if f_i == free_finger_id: continue
            l_wf[:,f_i * qnum : f_i * qnum + qnum] = l_wf_soln[:, i * qnum : i * qnum + qnum]
            i += 1

        # Number of interpolation points
        interp_n = 26

        # Linearly interpolate between each position waypoint (row) and force waypoint
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid-1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        #itp_vel = interp1d(row_ind_in, ft_vel, axis=0)
        itp_lwf = interp1d(row_ind_in, l_wf, axis=0)
        self.ft_pos_traj = itp_pos(row_coord_out)
        #self.ft_vel_traj = itp_vel(row_coord_out)
        self.l_wf_traj = itp_lwf(row_coord_out)

        # Linearly interpolate between each object pose
        # TODO: Does it make sense to linearly interpolate quaternions?
        itp_x_soln = interp1d(row_ind_in, self.x_soln, axis=0)
        self.x_traj = itp_x_soln(row_coord_out)

        # Zero-order hold for velocity waypoints
        self.ft_vel_traj = np.repeat(ft_vel, repeats=interp_n+1, axis=0)[:-interp_n, :]
        self.dx_traj = np.repeat(self.dx_soln, repeats=interp_n+1, axis=0)[:-interp_n, :]

    """
    Run trajectory optimization to move fingers to contact points on object
    """
    def set_traj_to_object(self, observation):
        self.traj_waypoint_counter = 0
        # First, set cp_params based on mode
        self.set_cp_params(observation)

        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        # Get joint positions
        current_position, _ = get_robot_position_velocity(observation)

        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        # Get list of desired fingertip positions
        cp_wf_list = c_utils.get_cp_pos_wf_from_cp_params(self.cp_params, obj_pose.position, obj_pose.orientation, use_obj_size_offset = True)

        # Deal with None fingertip_goal here
        # If cp_wf is None, set ft_goal to be  current ft position
        for i in range(len(cp_wf_list)):
            if cp_wf_list[i] is None:
                cp_wf_list[i] = current_ft_pos[i]

        ft_goal = np.asarray(cp_wf_list).flatten()
        self.ft_pos_traj, self.ft_vel_traj = self.run_finger_traj_opt(current_position, obj_pose, ft_goal)
        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None

    """
    Run traj opt to lower fingers to ground level
    """
    def set_traj_lower_finger(self, observation):
        self.traj_waypoint_counter = 0
        # First, set cp_params based on mode
        self.set_cp_params(observation)

        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        # Get joint positions
        current_position, _ = get_robot_position_velocity(observation)

        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        ft_goal = c_utils.get_pre_grasp_ft_goal(obj_pose, current_ft_pos, self.cp_params)

        self.ft_pos_traj, self.ft_vel_traj = self.run_finger_traj_opt(current_position, obj_pose, ft_goal)
        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None

    """
    Run trajectory optimization for fingers, given fingertip goal positions
    ft_goal: (9,) array of fingertip x,y,z goal positions in world frame
    """
    def run_finger_traj_opt(self, current_position, obj_pose, ft_goal):
        nGrid = self.finger_nlp.nGrid
        self.traj_waypoint_counter = 0

        ft_pos, ft_vel = c_utils.get_finger_waypoints(self.finger_nlp, ft_goal, current_position, obj_pose, npz_filepath = self.grasp_trajopt_filepath)

        print("FT_GOAL: {}".format(ft_goal))
        print(ft_pos[-1,:])

        # Number of interpolation points
        interp_n = 26

        # Linearly interpolate between each waypoint (row)
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid-1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        #itp_vel = interp1d(row_ind_in, ft_vel, axis=0)
        ft_pos_traj = itp_pos(row_coord_out)

        # Zero-order hold for velocity waypoints
        ft_vel_traj = np.repeat(ft_vel, repeats=interp_n+1, axis=0)[:-interp_n, :]

        return ft_pos_traj, ft_vel_traj

    def log_to_buffers(self, ft_pos_goal_list, ft_vel_goal_list,
                       cur_ft_pos, obj_pose, obj_vel, torque,
                       ft_des_force_wf=None):
        # LOGGING
        self.l_step_count.append(self.step_count)
        self.l_timestamp.append(time.time())
        self.l_desired_ft_pos.append(np.asarray(ft_pos_goal_list).flatten())
        self.l_desired_ft_vel.append(np.asarray(ft_vel_goal_list).flatten())
        self.l_actual_ft_pos.append(cur_ft_pos)
        self.l_observed_obj_pose.append(np.concatenate((obj_pose.position,obj_pose.orientation)))
        self.l_observed_filt_obj_pose.append(np.concatenate((self.filtered_obj_pose.position,self.filtered_obj_pose.orientation)))
        self.l_observed_obj_vel.append(obj_vel)
        self.l_desired_torque.append(np.asarray(torque))

        if self.x_traj is None:
            # Nan if there is no obj traj (during grasping)
            self.l_desired_obj_pose.append(np.ones(7) * np.nan)
        else:
            self.l_desired_obj_pose.append(self.x_traj[self.traj_waypoint_counter, :])
        if self.dx_traj is None:
            # Nan if there is no obj traj (during grasping)
            self.l_desired_obj_vel.append(np.ones(6) * np.nan)
        else:
            self.l_desired_obj_vel.append(self.dx_traj[self.traj_waypoint_counter, :])
        if ft_des_force_wf is None:
            # Nan if no desired ft forces (during grasping)
            self.l_desired_ft_force.append(np.ones(9) * np.nan)
        else:
            self.l_desired_ft_force.append(ft_des_force_wf)
        return

    """
    Replans trajectory according to TrajMode, and sets self.traj_waypoint_counter
    """
    def plan_trajectory(self, observation):
        if self.mode == TrajMode.RESET:
            self.set_traj_lower_finger(observation)
            self.mode = TrajMode.PRE_TRAJ_LOWER
        elif self.mode == TrajMode.PRE_TRAJ_LOWER:
            self.set_traj_to_object(observation)
            self.mode = TrajMode.PRE_TRAJ_REACH
        elif self.mode == TrajMode.PRE_TRAJ_REACH:
            self.set_traj_lift_object(observation, nGrid=50, dt=0.08)
            self.mode = TrajMode.REPOSE
        elif self.mode == TrajMode.REPOSE:
            print("ERROR: should not reach this case")

        self.traj_waypoint_counter = 0
        return

    # TODO: What about when object observations are noisy???
    def get_obj_vel(self, cur_obj_pose, cur_step_time):
        dt = cur_step_time - self.prev_step_time

        if dt == 0:
            return self.filt_vel

        obj_vel_position = (cur_obj_pose.position - self.prev_obj_pose.position) / dt

        # TODO: verify that we are getting angular velocities from quaternions correctly
        #cur_R = Rotation.from_quat(cur_obj_pose.orientation)
        #prev_R = Rotation.from_quat(self.prev_obj_pose.orientation)
        #delta_R = cur_R * prev_R.inv()
        #obj_vel_quat = delta_R.as_quat() / dt
        obj_vel_quat = (cur_obj_pose.orientation - self.prev_obj_pose.orientation) / dt
        M = c_utils.get_dquat_to_dtheta_matrix(self.prev_obj_pose.orientation) # from Paul Mitiguy dynamics notes
        obj_vel_theta = 2 * M @ obj_vel_quat
        #obj_vel_theta = np.zeros(obj_vel_theta.shape)

        cur_vel = np.concatenate((obj_vel_position, obj_vel_theta))

        # Set previous obj_pose and step_time to current values
        self.prev_obj_pose = cur_obj_pose
        self.prev_step_time = cur_step_time
        self.prev_vel = cur_vel

        # filter the velocity
        theta = 0.1
        filt_vel = (1-theta) * self.filt_vel + theta * cur_vel
        self.filt_vel = filt_vel.copy()

        # Log obj_vel_quat for debugging
        if self.DEBUG:
            self.l_dquat.append(obj_vel_quat)

        return filt_vel
        return cur_vel

    def predict(self, full_observation):
        self.step_count += 1
        observation = full_observation['observation']
        current_position, current_velocity = observation['position'], observation['velocity']

        # Get object pose
        obj_pose = get_pose_from_observation(full_observation)
        # Filter object pose
        self.set_filtered_pose_from_observation(full_observation)

        # Estimate object velocity based on previous and current object pose
        # TODO: this might cause an issue if observed object poses are the same across steps?
        if osp.exists("/output"):
            timestamp = full_observation["cam0_timestamp"]
        else:
            timestamp = full_observation["cam0_timestamp"] / 1000
        print("Cam0_timestamp: {}".format(timestamp))
        obj_vel = self.get_obj_vel(self.filtered_obj_pose, timestamp)

        # Get current fingertip position
        cur_ft_pos = self.get_fingertip_pos_wf(current_position)
        cur_ft_pos = np.asarray(cur_ft_pos).flatten()

        if self.traj_waypoint_counter == self.ft_pos_traj.shape[0]:
            # TODO: currently will redo the last waypoint after reaching end of trajectory
            self.plan_trajectory(full_observation)

        ft_pos_goal_list = []
        ft_vel_goal_list = []
        # If object is grasped, transform cp_wf to ft_wf
        if self.mode == TrajMode.REPOSE:
            H_list = c_utils.get_ft_R(current_position)

        for f_i in range(3):
            new_pos = self.ft_pos_traj[self.traj_waypoint_counter, f_i*3:f_i*3+3]
            new_vel = self.ft_vel_traj[self.traj_waypoint_counter, f_i*3:f_i*3+3]
            #print(f_i)

            #print(new_pos)
            if self.mode == TrajMode.REPOSE:
                H = H_list[f_i]
                temp = H @ np.array([0, 0, 0.0095])
                #print("temp: {}".format(temp))
                new_pos = np.array(new_pos) + temp[:3]
                new_pos = new_pos.tolist()
            #print(new_pos)

            ft_pos_goal_list.append(new_pos)
            ft_vel_goal_list.append(new_vel)
        # if self.mode == TrajMode.REPOSE:
           # quit()

        # If in REPOSE, get fingertip forces in world frame
        if self.mode == TrajMode.REPOSE:
            ft_des_force_wf, W = c_utils.get_ft_forces(self.x_traj[self.traj_waypoint_counter, :],
                                 self.dx_traj[self.traj_waypoint_counter, :],
                                 obj_pose, obj_vel, self.KP_OBJ, self.KV_OBJ,
                                 self.cp_params)
            ft_des_force_wf = np.asarray(ft_des_force_wf).flatten()
            # ft_des_force_wf = self.l_wf_traj[self.traj_waypoint_counter, :]

            #if self.DEBUG:
            #    self.l_desired_obj_w.append(W.flatten())

        else:
            ft_des_force_wf = None

        # Compute torque with impedance controller, and clip
        if self.mode == TrajMode.REPOSE:
            KP = self.KP_REPOSE
            KV = self.KV_REPOSE
        else:
            KP = self.KP
            KV = self.KV

        torque = c_utils.impedance_controller(ft_pos_goal_list,
                                              ft_vel_goal_list,
                                              current_position, current_velocity,
                                              self.custom_pinocchio_utils,
                                              tip_forces_wf=ft_des_force_wf,
                                              Kp = KP, Kv = KV)
        torque = np.clip(torque, self.action_space.low, self.action_space.high)

        self.log_to_buffers(ft_pos_goal_list, ft_vel_goal_list,
                            cur_ft_pos, obj_pose, obj_vel, torque,
                            ft_des_force_wf)
        # always increment traj_waypoint_counter UNLESS in repose mode and have reached final waypoint
        if not (self.mode == TrajMode.REPOSE and self.traj_waypoint_counter == self.ft_pos_traj.shape[0] - 1):
            self.traj_waypoint_counter += 1
        return torque

    """
    Get fingertip positions in world frame given current joint q
    """
    def get_fingertip_pos_wf(self, current_q):
        fingertip_pos_wf = self.custom_pinocchio_utils.forward_kinematics(current_q)
        return fingertip_pos_wf

    """
    """
    def set_filtered_pose_from_observation(self, observation, theta=0.01):
        new_pose = get_pose_from_observation(observation)

        f_p = (1-theta) * self.filtered_obj_pose.position + theta * new_pose.position
        f_o = (1-theta) * self.filtered_obj_pose.orientation + theta * new_pose.orientation

        filt_pose = move_cube.Pose(position=f_p, orientation=f_o)
        self.filtered_obj_pose = filt_pose

        return filt_pose



class HierarchicalControllerPolicy:
    DIST_THRESH = 0.09
    ORI_THRESH = np.pi / 6

    RESET_TIME_LIMIT = 50
    RL_RETRY_STEPS = 70
    MAX_RETRIES = 3

    default_robot_position = TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, load_dir='', start_mode=PolicyMode.RL_PUSH,
                 difficulty=1, deterministic=True, debug_waypoints=False):
        self.full_action_space = action_space
        action_space = action_space['torque']
        self.impedance_controller = ImpedanceControllerPolicy(
                action_space, initial_pose, goal_pose, npz_file, debug_waypoints=debug_waypoints,
                difficulty = difficulty)
        self.load_policy(load_dir, deterministic)
        self.start_mode = start_mode
        self._platform = None
        self.steps_from_reset = 0
        self.step_count = self.rl_start_step = 0
        self.traj_initialized = False
        self.rl_retries = int(self.start_mode == PolicyMode.RL_PUSH)
        self.difficulty = difficulty

    def reset_policy(self, observation, platform=None):
        self.mode = self.start_mode
        self.traj_initialized = False
        self.steps_from_reset = self.step_count = self.rl_start_step = 0
        if platform:
            self._platform = platform
        self.impedance_controller.reset_policy(observation, platform)

    @property
    def platform(self):
        assert self._platform is not None, 'HierarchicalControlPolicy.platform is not set'
        return self._platform

    @platform.setter
    def platform(self, platform):
        assert platform is not None, 'platform is not yet initialized'
        self._platform = platform
        self.impedance_controller.platform = platform

    def load_policy(self, load_dir, deterministic=False):
        self.observation_names = []
        if not load_dir:
            self.rl_frameskip = 1
            self.rl_observation_space = None
            self.rl_policy = lambda obs: self.impedance_controller.predict(obs)
        elif osp.exists(load_dir) and 'pyt_save' in os.listdir(load_dir):
            self.load_spinup_policy(load_dir, deterministic=deterministic)
        else:
            self.load_sb_policy(load_dir)

    def load_sb_policy(self, load_dir):
        # loads make_env, make_reorient_env, and make_model helpers
        assert 'HER-SAC' in load_dir, 'only configured HER-SAC policies so far'
        if '_push' in load_dir:
            self.rl_env = sb_utils.make_env()
        else:
            self.rl_env = sb_utils.make_reorient_env()
        self.rl_frameskip = self.rl_env.unwrapped.frameskip
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        self.sb_policy = sb_utils.make_her_sac_model(None, None)
        self.sb_policy.load(load_dir)
        self.rl_policy = lambda obs: self.sb_policy.predict(obs)[0]

    def load_spinup_policy(self, load_dir, load_itr='last', deterministic=False):
        self.rl_env, self.rl_policy = load_policy_and_env(load_dir, load_itr, deterministic)
        if self.rl_env:
            self.rl_frameskip = self.rl_env.frameskip
        else:
            self.rl_frameskip = 10
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        print('loaded policy from {}'.format(load_dir))

    def activate_rl(self, obj_pose):
        if self.start_mode != PolicyMode.RL_PUSH or self.rl_retries == self.MAX_RETRIES:
            if self.rl_retries == self.MAX_RETRIES and self.difficulty == 4:
                self.difficulty = 3
            return False
        return np.linalg.norm(obj_pose.position[:2] - np.zeros(2)) > self.DIST_THRESH

    def initialize_traj_opt(self, observation):
        obj_pose = get_pose_from_observation(observation)
        goal_pose = get_pose_from_observation(observation, goal_pose=True)

        # TODO: check orientation error
        if (self.activate_rl(obj_pose) and
            self.start_mode == PolicyMode.RL_PUSH and
            self.mode != PolicyMode.RESET):
            if self.mode != PolicyMode.RL_PUSH:
                self.mode = PolicyMode.RL_PUSH
                self.rl_start_step = self.step_count
            elif self.step_count - self.rl_start_step == self.RL_RETRY_STEPS:
                self.mode = PolicyMode.RESET
            return False
        elif self.mode == PolicyMode.RL_PUSH:
            if self.step_count > 0:
                self.mode = PolicyMode.RESET
                return False
            else: # skips reset if starting at RL_PUSH
                self.mode = PolicyMode.TRAJ_OPT
                return True
        elif (self.mode == PolicyMode.RESET and
              (self.steps_from_reset >= self.RESET_TIME_LIMIT and
               obj_pose.position[2] < 0.034)):
            self.steps_from_reset = 0
            if self.activate_rl(obj_pose):
                self.rl_retries += 1
                self.mode = PolicyMode.RL_PUSH
                self.rl_start_step = self.step_count
                return False
            else:
                self.mode = PolicyMode.TRAJ_OPT
                return True
        elif self.mode == PolicyMode.TRAJ_OPT:
            return True
        return True

    def set_waypoints(self, observation):
        if self.mode == PolicyMode.TRAJ_OPT:
            init_pose = get_pose_from_observation(observation)
            goal_pose = get_pose_from_observation(observation, goal_pose=True)
            if self.difficulty == 4:
                self.impedance_controller.set_init_goal(
                        init_pose, goal_pose, flip=flip_needed(init_pose, goal_pose))
            else:
                self.impedance_controller.set_init_goal(init_pose, goal_pose)

            #self.impedance_controller.set_traj_to_object(observation)
            self.traj_initialized = True  # pre_traj_wp are initialized
            self.mode = PolicyMode.IMPEDANCE

    def reset_action(self, observation):
        robot_position = observation['observation']['position']
        time_limit_step = self.RESET_TIME_LIMIT // 3
        if self.steps_from_reset < time_limit_step:  # middle
            robot_position[1::3] = self.default_robot_position[1::3]
        elif time_limit_step <= self.steps_from_reset < 2*time_limit_step:  # tip
            robot_position[2::3] = self.default_robot_position[2::3]
        else:  # base
            robot_position[::3] = self.default_robot_position[::3]
        self.steps_from_reset += 1
        return robot_position

    def predict(self, observation):
        if not self.traj_initialized and self.initialize_traj_opt(observation['impedance']):
            self.set_waypoints(observation['impedance'])

        if self.mode == PolicyMode.RL_PUSH and self.rl_observation_space is not None:
            ac = self.rl_policy(observation['rl'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
        elif self.mode == PolicyMode.RESET:
            ac = self.reset_action(observation['impedance'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
        elif self.mode == PolicyMode.IMPEDANCE:
            ac = self.impedance_controller.predict(observation['impedance'])
        else:
            assert False, 'use a different start mode, started with: {}'.format(self.start_mode)
        self.step_count += 1
        return ac


class ResidualControllerPolicy(HierarchicalControllerPolicy):
    DIST_THRESH = 0.09
    ORI_THRESH = np.pi / 6
    default_robot_position = TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, start_mode=PolicyMode.RL_PUSH, difficulty=1,
                 rl_torque=True, rl_tip_pos=False, rl_cp_params=False,
                 debug_waypoints=False):
        super(action_space, initial_pose, goal_pose, npz_file, load_dir='',
              start_mode=PolicyMode.TRAJ_OPT, difficulty=difficulty, deterministic=True,
              debug_waypoints=debug_waypoints)
        self.rl_torque = rl_torque
        self.rl_tip_pos = rl_tip_pos
        self.rl_cp_params = rl_cp_params

    def process_observation_rl(self, observation, torque):
        if self.rl_torque:
            observation = np.concatenate([observation, torque])
        return observation

    def predict(self, observation):
        if not self.traj_initialized and self.initialize_traj_opt(observation['impedance']):
            self.set_waypoints(observation['impedance'])
        torque = self.impedance_controller.predict(observation['impedance'])
        rl_obs = self.process_observation_rl(observation['rl'], torque)
        if self.rl_torque:
            torque += self.rl_policy(rl_obs)
        return torque


def get_pose_from_observation(observation, goal_pose=False):
    use_filtered = osp.exists("/output") # If using backend, use filtered object pose

    if goal_pose:
        key = "desired_goal"
    else:
        if use_filtered:
            key = "filtered_achieved_goal"
        else:
            key = "achieved_goal"
    return move_cube.Pose.from_dict(observation[key])

def flip_needed(init_pose, goal_pose):
    return (c_utils.get_closest_ground_face(init_pose) !=
            c_utils.get_closest_ground_face(goal_pose))

def get_robot_position_velocity(observation):
    observation = observation['observation']
    return observation['position'], observation['velocity']


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
    env = state['env']

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            if deterministic:
                action = model.pi(x)[0].mean.numpy()
            else:
                action = model.act(x)
        return action

    return get_action
