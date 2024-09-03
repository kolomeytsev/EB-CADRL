import time
import pandas as pd
import multiprocessing
import logging
import configparser
import os
import torch
import gym

from rl.policy.sarl import SARL
from rl.test import parse_arguments
from simulator.agents.robot import Robot
from simulator.utils.info import *


PHASE = "test"


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0


def run_episode_wrapper(args_tuple):
    """Wrapper function to unpack arguments."""
    return run_episode(*args_tuple)


def run_episode(args, env_config, policy_config_file, model_weights_path, episode):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info("Using device: %s", device)

    policy = SARL()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if model_weights_path is None:
            exit(1)
        policy.get_model().load_state_dict(torch.load(model_weights_path))

    env = gym.make("EntityBasedCollisionAvoidance-v0")
    env.configure(env_config)
    if args.square:
        env.test_sim_adult = "square_crossing"
    if args.circle:
        env.test_sim_adult = "circle_crossing"
    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    env.set_robot(robot)
    policy.set_phase(PHASE)
    policy.set_device(device)

    # policy.set_env(env)

    success = 0

    collision = 0
    collision_adult = 0
    collision_bicycle = 0
    collision_child = 0
    collision_obstacle = 0

    timeout = 0
    too_close = 0
    min_dist = []
    ob, local_map = env.reset(PHASE, scene_number=episode)

    done = False
    states = []
    actions = []
    rewards = []
    torch.set_num_threads(1)
    dmin_adult = []
    dmin_bicycle = []
    dmin_child = []
    while not done:
        action = robot.act(ob, local_map=local_map, env=env)
        ob, local_map, reward, done, info = env.step(action)
        states.append(robot.policy.last_state)
        actions.append(action)
        rewards.append(reward)

        if info.dmin_adult is not None:
            dmin_adult.append(info.dmin_adult)
        if info.dmin_bicycle is not None:
            dmin_bicycle.append(info.dmin_bicycle)
        if info.dmin_child is not None:
            dmin_child.append(info.dmin_child)

        if isinstance(info, Danger):
            too_close += 1
            min_dist.append(info.min_dist)

    if isinstance(info, ReachGoal):
        success += 1
    elif isinstance(info, Collision):
        collision += 1
    elif isinstance(info, CollisionChild):
        collision_child += 1
    elif isinstance(info, CollisionAdult):
        collision_adult += 1
    elif isinstance(info, CollisionBicycle):
        collision_bicycle += 1
    elif isinstance(info, CollisionObstacle):
        collision_obstacle += 1
    elif isinstance(info, Timeout):
        timeout += 1
    else:
        raise ValueError("Invalid end signal from environment")

    cumulative_reward = sum(
        [
            pow(policy.gamma, t * robot.time_step * robot.v_pref) * reward
            for t, reward in enumerate(rewards)
        ]
    )
    result = {
        "episode": episode,
        "time": env.global_time,
        "reward": cumulative_reward,
        "success": success,
        "collision": collision,
        "collision_child": collision_child,
        "collision_adult": collision_adult,
        "collision_bicycle": collision_bicycle,
        "collision_obstacle": collision_obstacle,
        "timeout": timeout,
        "too_close": too_close,
        "min_dist": min_dist,
        "dist_to_goal": info.dist_to_goal,
        "dmin_adult": dmin_adult,
        "dmin_bicycle": dmin_bicycle,
        "dmin_child": dmin_child,
    }
    return result


def main():
    args = parse_arguments()
    if args.csv == "":
        exit(1)

    model_weights_path = None
    if args.model_dir is not None:
        env_config_file = os.path.join(
            args.model_dir, os.path.basename(args.env_config)
        )
        policy_config_file = os.path.join(
            args.model_dir, os.path.basename(args.policy_config)
        )
        model_weights_path = os.path.join(args.model_dir, args.model_name)
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    if args.model_path is not None:
        model_weights_path = args.model_path

    # configure logging and device
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)

    t0 = time.time()
    episodes = range(args.start, args.end)
    args_list = [
        (args, env_config, policy_config_file, model_weights_path, episode)
        for episode in episodes
    ]
    # results = []
    # for args in args_list:
    #     results.append(run_episode(*args))
    print("multiprocessing.cpu_count():", multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(run_episode, args_list)
    df = pd.DataFrame(results)
    df.to_csv(args.csv)
    print("time passed: ", time.time() - t0)


if __name__ == "__main__":
    main()
