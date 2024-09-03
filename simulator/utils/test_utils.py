import configparser
import torch
import gym
from crowd_nav.policy.policy_factory import policy_factory
from simulator.agents.robot import Robot


def configure_env_policy_robot(env_config_path, policy_config_path, model_path=None, phase="test",
                               device="cpu", policy="sarl", env_name="CrowdSimStatic-v0"):
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_path)
    env = gym.make(env_name)
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policy = policy_factory[policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_path)
    policy.configure(policy_config)
    if model_path is not None:
        policy.get_model().load_state_dict(torch.load(model_path))

    robot.set_policy(policy)
    policy.set_phase(phase)
    policy.set_device(device)
    # policy.set_env(env)

    return env, policy, robot
