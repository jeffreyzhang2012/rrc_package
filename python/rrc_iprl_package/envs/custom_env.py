"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym
import pybullet

from gym import wrappers
from gym import ObservationWrapper

import robot_interfaces
import robot_fingers
import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube
from rrc_iprl_package import cube_env
from rrc_simulation.control_env import PolicyMode
from rrc_simulation.control_policy import HierarchicalControllerPolicy


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = move_cube._CUBE_WIDTH / 5
ORI_THRESH = np.pi / 8
REW_BONUS = 1
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


def reset_camera():
    camera_pos = (0.,0.2,-0.2)
    camera_dist = 1.0
    pitch = -45.
    yaw = 0.
    if pybullet.isConnected() != 0:
        pybullet.resetDebugVisualizerCamera(cameraDistance=camera_dist,
                                    cameraYaw=yaw,
                                    cameraPitch=pitch,
                                    cameraTargetPosition=camera_pos)


class PushCubeEnv(gym.Env):
    observation_names = ["robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position", 
            "action"]

    def __init__(
        self,
        cube_goal_pose=None,
        goal_difficulty=1,
        action_type=ActionType.POSITION,
        frameskip=1,
        visualization=False,
        ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose. If no initializer is provided, we will initialize in a way
                which is be helpful for learning.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================
        self.goal = cube_goal_pose
        self.info = {'difficulty': goal_difficulty}
        self.visualization = visualization

        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None
        self._prev_action = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )

        # verify that the given goal pose is contained in the cube state space
        if not object_state_space.contains(self.goal):
            raise ValueError("Invalid goal pose.")

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")
        obs_spaces = {
                "robot_position": robot_position_space,
                "robot_velocity": robot_velocity_space,
                "robot_torque": robot_torque_space,
                "action": self.action_space,
                "goal_position": object_state_space.spaces['position'],
                "goal_orientation": object_state_space.spaces['orientation'],
                "object_position": object_state_space.spaces['position'],
                "object_orientation": object_state_space.spaces['orientation'],
            }

        self.observation_space = gym.spaces.Dict({k:obs_spaces[k] for k in observation_names})

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def _reset_platform_frontend(self):
        """Reset the platform frontend."""
        # reset is not really possible
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformFrontend()

    def _reset_direct_simulation(self):
        """Reset direct simulation.

        With this the env can be used without backend.
        """
        # set this to false to disable pyBullet's simulation
        visualization = True

        # reset simulation
        del self.platform

        # initialize simulation
        initial_object_pose = move_cube.sample_goal(difficulty=-1)
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=visualization,
            initial_object_pose=initial_object_pose,
        )

        # visualize the goal
        if visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=0.065,
                position=self.goal["position"],
                orientation=self.goal["orientation"],
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )
            reset_camera()

    def reset(self):
        # reset simulation
        del self.platform
        if self._sim_backend:
            self._reset_platform_frontend()
        else: 
            self._reset_direct_simulation()

        self.info = {"difficulty": self.initializer.difficulty}
        self.step_count = 0
        observation, _, _, _ = self.step(self._initial_action)
        return observation

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_torque": robot_observation.torque,
            "robot_tip_positions": robot_tip_positions,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
            "action": action
        }
        return {k: observation[k] for k in self.observation_names}

    @staticmethod
    def _compute_reward(previous_observation, observation):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = 500 * reward_term_1 + 250 * reward_term_2
        return reward

    def step(self, action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            previous_observation = self._create_observation(t, self._prev_action)
            observation = self._create_observation(t + 1, action)

            reward += self._compute_reward(
                previous_observation=previous_observation,
                observation=observation,
            )

        is_done = self.step_count == move_cube.episode_length
        if is_done and isinstance(self.initializer, CurriculumInitializer):
            goal_pose = self.goal
            if not isinstance(goal_pose, move_cube.Pose):
                goal_pose = move_cube.Pose.from_dict(goal_pose)
            object_pose = move_cube.Pose.from_dict(dict(
                position=observation['object_position'].flatten(),
                orientation=observation['object_orientation'].flatten()))
            self.initializer.update_initializer(object_pose, goal_pose)

        self._prev_action = action
        return observation, reward, is_done, self.info


class ResidualPolicyWrapper(ObservationWrapper):
    def __init__(self, env, policy):
        assert isinstance(env.unwrapped, cube_env.CubeEnv), 'env expects type CubeEnv'
        self.env = env
        self.reward_range = self.env.reward_range
        # set observation_space and action_space below
        spaces = TriFingerPlatform.spaces
        self._action_space = gym.spaces.Dict({
            'torque': spaces.robot_torque.gym, 'position': spaces.robot_position.gym})
        self.set_policy(policy)

    @property
    def impedance_control_mode(self):
        return (self.mode == PolicyMode.IMPEDANCE or
                (self.mode == PolicyMode.RL_PUSH and
                 self.rl_observation_space is None))

    @property
    def action_space(self):
        if self.impedance_control_mode:
            return self._action_space['torque']
        else:
            return self._action_space['position']

    @property
    def action_type(self):
        if self.impedance_control_mode:
            return ActionType.TORQUE
        else:
            return ActionType.POSITION

    @property
    def mode(self):
        assert self.policy, 'Need to first call self.set_policy() to access mode'
        return self.policy.mode

    @property
    def frameskip(self):
        if self.mode == PolicyMode.RL_PUSH:
            return self.policy.rl_frameskip
        return 1

    def set_policy(self, policy):
        self.policy = policy
        if policy:
            self.rl_observation_names = policy.observation_names
            self.rl_observation_space = policy.rl_observation_space
            obs_dict = {'impedance': self.env.observation_space}
            if self.rl_observation_space:
                obs_dict['rl'] = self.rl_observation_space
            self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, observation):
        observation_imp = self.process_observation_impedance(observation)
        obs_dict = {'impedance': observation_imp}
        if 'rl' in self.observation_space.spaces:
            observation_rl = self.process_observation_rl(observation)
            obs_dict['rl'] = observation_rl
        return obs_dict

    def process_observation_residual(self, observation):
        return observation

    def process_observation_rl(self, observation):
        t = self.step_count
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
            "goal_object_orientation": self.goal["orientation"],
        }
        observation = np.concatenate([observation[k].flatten() for k in self.rl_observation_names])
        return observation

    def process_observation_impedance(self, observation):
        return observation

    def reset(self):
        obs = super(ResidualPolicyWrapper, self).reset()
        self.policy.platform = self.env.unwrapped.platform
        self.policy.reset_policy()
        self.step_count = 0
        return obs

    def _step(self, action):
        if self.unwrapped.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.unwrapped.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            observation = self.unwrapped._create_observation(t + 1)

            reward += self.unwrapped.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.unwrapped.info,
            )

        is_done = self.step_count == move_cube.episode_length

        return observation, reward, is_done, self.env.info

    def _gym_action_to_robot_action(self, gym_action):
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action):
        # CubeEnv handles gym_action_to_robot_action
        #print(self.mode)
        if self.mode == PolicyMode.RL_PUSH:
            self.unwrapped.frameskip = self.policy.rl_frameskip
        else:
            self.unwrapped.frameskip = 1

        obs, r, d, i = self._step(action)
        obs = self.observation(obs)
        return obs, r, d, i
