import torch
import logging
import configparser
import gym
from simulator.agents.robot import Robot
from rl.policy.policy_factory import policy_factory


def configure_policy(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("Using device: %s", device)

    policy = policy_factory[args.policy]()
    if not policy.trainable:
        raise Exception("Policy has to be trainable")
    if args.policy_config is None:
        raise Exception("Policy config has to be specified for a trainable network")
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)
    return policy


def configure_environment_and_robot(args, env_name):
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make(env_name)
    env.configure(env_config)
    robot = Robot(env_config, "robot")
    env.set_robot(robot)
    return robot, env
