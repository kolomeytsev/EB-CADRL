import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
from rl.utils.trainer import Trainer
from rl.utils.memory import ReplayMemory
from rl.utils.explorer import Explorer
from rl.utils.parallel_explorer import ParallelExplorer
from rl.policy.policy_factory import policy_factory
from rl.utils.utils import configure_policy, configure_environment_and_robot
import time
import gc


VAL_EPISODE_START = 100000
PROCESSES_NUM = 8


def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env_configs/env.config')
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--policy_config', type=str, default='configs/policy_configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train_configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_iteration', type=int, default=0)
    parser.add_argument('--end_iteration', type=int, default=0)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    return parser, args


def configure_paths(args):
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.resume and args.resume_iteration > 0:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y':
                print('Output directory was overwriten.')
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
                args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
                args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
                args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))

    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)


def configure_logging(args):
    log_file = os.path.join(args.output_dir, 'output.log')
    # configure logging
    mode = 'a' if (args.resume or args.resume_iteration != 0) else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    # level = logging.ERROR if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


def run_imitation_learning(trainer, explorer, robot, policy, args, memory, model, train_config):
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')

    if args.resume and args.resume_iteration != 0:
        rl_weight_file = os.path.join(args.output_dir, f'rl_model_{args.resume_iteration}.pth')
        print("rl_weight_file: ", rl_weight_file)
        if not os.path.exists(rl_weight_file):
            logging.error(f'RL weights does not exist: {rl_weight_file}')
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        t0 = time.time()
        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_optimizer(il_learning_rate)
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        torch.save(model.state_dict(), il_weight_file)
        logging.info('Finish imitation learning. Weights saved. Time passed: %f', time.time() - t0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)


def save_rl_model(model, args, episode):
    logging.info('Save model episode: %d', episode)
    rl_weight_file = os.path.join(args.output_dir, 'rl_model_' + str(episode) + '.pth')
    torch.save(model.state_dict(), rl_weight_file)


def run_train(args, policy, env, robot):
    # read training parameters
    if args.train_config is None:
        raise Exception("Train config has to be specified for a trainable network")

    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
    train_batches = train_config.getint('train', 'train_batches')
    train_episodes = train_config.getint('train', 'train_episodes')
    # sample_episodes = train_config.getint('train', 'sample_episodes')
    target_update_interval = train_config.getint('train', 'target_update_interval')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    capacity = train_config.getint('train', 'capacity')
    epsilon_start = train_config.getfloat('train', 'epsilon_start')
    epsilon_end = train_config.getfloat('train', 'epsilon_end')
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')
    optimizer_algorithm = train_config.get('train', 'optimizer_algorithm')

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    model.share_memory()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, policy.device, batch_size, optimizer_algorithm)
    explorer = Explorer(env, robot, policy.device, memory, policy.gamma, target_policy=policy)
    parallel_explorer = ParallelExplorer(env, robot, policy.device, PROCESSES_NUM,
                                         memory, policy.gamma, target_policy=policy)

    # imitation learning
    run_imitation_learning(trainer, explorer, robot, policy, args, memory, model, train_config)
    explorer.update_target_model(model)
    parallel_explorer.update_target_model(model)

    # reinforcement learning
    # policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_optimizer(rl_learning_rate)
    # fill the memory pool with some RL experience
    # if args.resume:
    #     robot.policy.set_epsilon(epsilon_end)
    #     explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
    #     logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    episode = args.resume_iteration if args.resume and args.resume_iteration != 0 else 0
    while episode < train_episodes:
        t0 = time.time()
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end

        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            try:
                parallel_explorer.run_k_episodes_parallel(
                    args, VAL_EPISODE_START, VAL_EPISODE_START + env_config.getint('env', 'val_size'), 'val')
            except RuntimeError as e:
                logging.info('Caught RuntimeError in run_k_episodes_parallel val: {}'.format(str(e)))

        # sample k episodes into memory and optimize over the generated memory
        try:
            parallel_explorer.run_k_episodes_parallel(
                args, episode, episode + PROCESSES_NUM, 'train', update_memory=True)
            trainer.optimize_batch(train_batches)
            episode += PROCESSES_NUM
        except RuntimeError as e:
            logging.info('Caught RuntimeError in run_k_episodes_parallel train: {}'.format(str(e)))

        if episode % target_update_interval == 0:
            try:
                parallel_explorer.update_target_model(model)
            except RuntimeError as e:
                logging.info('Caught RuntimeError in update_target_model: {}'.format(str(e)))

        if episode != 0 and episode % checkpoint_interval == 0:
            try:
                save_rl_model(model, args, episode)
            except RuntimeError as e:
                logging.info('Caught RuntimeError in save_rl_model: {}'.format(str(e)))

        logging.info('Time passed: %f', time.time() - t0)

        if args.end_iteration > 0 and episode >= args.end_iteration:
            logging.info('Ending train on episode: {episode}, end_iteration: {args.end_iteration}')
            break

    save_rl_model(model, args, episode)
    return parallel_explorer, episode


def get_max_iteration(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    iterations = []
    for x in onlyfiles:
        try:
            if len(x) > 3 and x[-4:] == '.pth':
                iterations.append(int(x[:-4].split('_')[2]))
        except Exception as e:
            print("Exception", e)

    max_iteration = None
    if len(iterations):
        max_iteration = max(iterations)
    return max_iteration


def main():
    parser, args = parse_arguments()
    configure_paths(args)
    configure_logging(args)

    policy = configure_policy(args)
    robot, env = configure_environment_and_robot(args, 'EntityBasedCollisionAvoidance-v0')

    t0 = time.time()
    logging.info(f"Run train with arguments: {' '.join(f'{k}={v}' for k, v in vars(args).items())}")
    explorer, episode = run_train(args, policy, env, robot)
    print("Train finished, time taken:", time.time() - t0)

    # TODO: enable final test
    # t0 = time.time()
    # explorer.run_k_episodes(env.scene.case_size['test'], 'test', episode=episode)
    # print("Final test finished, time taken:", time.time() - t0)


if __name__ == '__main__':
    main()
