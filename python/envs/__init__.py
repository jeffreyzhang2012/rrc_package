import gym
from gym.envs.registration import register

registered_envs = [spec.id for spec in gym.envs.registry.all()]

if "real_robot_challenge_phase_2-v0" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v0",
        entry_point="rrc_iprl_package.envs.cube_env:RealRobotCubeEnv",
    )

if "real_robot_challenge_phase_2-v1" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v1",
        entry_point="rrc_iprl_package.envs.cube_env:CubeEnv",
    )

if "real_robot_challenge_phase_2-v2" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v2",
        entry_point="rrc_iprl_package.envs.custom_env:PushCubeEnv",
    )

