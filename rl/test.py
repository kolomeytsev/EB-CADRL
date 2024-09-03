import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from rl.utils.explorer import Explorer
from rl.policy.policy_factory import policy_factory
from simulator.agents.robot import Robot
from simulator.policy.orca import ORCA


PHASE  = 'test'


def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env_configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy_configs/policy.config')
    parser.add_argument('--csv', type=str, default="")
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    model_weights_path = None
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights_path = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights_path = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights_path = os.path.join(args.model_dir, 'rl_model_50000.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    if args.model_path is not None:
        model_weights_path = args.model_path

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if model_weights_path is None:
            raise Exception('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights_path))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('EntityBasedCollisionAvoidance-v0')
    env.configure(env_config)
    if args.square:
        env.scene.test_sim_adult = 'square_crossing'
    if args.circle:
        env.scene.test_sim_adult = 'circle_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=policy.gamma)

    policy.set_phase(PHASE)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    # policy.set_env(env)
    robot.print_info()
    if args.visualize:
        # ob = env.reset(PHASE, args.test_case)
        if robot.policy.name == 'ORCA':
            ob, global_map, local_map = env.reset(PHASE, args.test_case, scene_number=args.test_case)  # TODO: use global_map
        else:
            ob, local_map = env.reset(PHASE, args.test_case, scene_number=args.test_case)

        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob, local_map=local_map, env=env)
            ob, local_map, reward, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        # if robot.visible and info == 'reach goal':
        #     adult_times = env.get_adult_times()
        #     logging.info('Average time for adults to reach goal: %.2f', sum(adult_times) / len(adult_times))
    else:
        explorer.run_k_episodes(env.scene.case_size[PHASE], PHASE, print_failure=True)


if __name__ == '__main__':
    main()
