import pprint
from simulator.utils.info import *
from rl.train import configure_environment_and_robot, run_train
import logging
import sys
import os
import torch
import time
import configparser
from rl.policy.policy_factory import policy_factory
import collections
import tempfile


Params = collections.namedtuple(
    "Params",
    [
        "env_config",
        "policy",
        "policy_config",
        "train_config",
        "output_dir",
        "resume",
        "gpu",
        "debug",
        "end_iteration",
    ],
)


def configure_logging(params):
    log_file = os.path.join(params.output_dir, "output.log")
    # configure logging
    mode = "a" if params.resume else "w"
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not params.debug else logging.DEBUG
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_basic_train_with_output_dir(output_dir, env_config):
    params = Params(
        env_config=env_config,
        policy="sarl",
        policy_config="configs/test_configs/test_policy_configs/policy.config",
        train_config="configs/test_configs/test_train_configs/test_train.config",
        output_dir=output_dir,
        resume=False,
        gpu=False,
        debug=False,
        end_iteration=0,
    )
    configure_logging(params)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and params.gpu else "cpu"
    )
    logging.info("Using device: %s", device)

    policy = policy_factory[params.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(params.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)

    robot, env = configure_environment_and_robot(
        params, "EntityBasedCollisionAvoidance-v0"
    )

    t0 = time.time()
    explorer, episode = run_train(params, policy, env, robot)
    print("Train finished, time taken:", time.time() - t0)

    # TODO: enable final test
    # t0 = time.time()
    # explorer.run_k_episodes(env.scene.case_size['test'], 'test', episode=episode)
    # print("Final test finished, time taken:", time.time() - t0)
    success = True
    result = {"info": "Train was successful"}
    return success, result


def run_basic_train():
    env_config = "configs/test_configs/test_env_configs/env_adults_3_bikes_3_child_3_static_3_fast_train.config"
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        print(tmp_output_dir)
        return run_basic_train_with_output_dir(tmp_output_dir, env_config)


if __name__ == "__main__":
    res, info = run_basic_train()
    if res:
        print("Test passed!")
    else:
        print("Error!")
    pprint.pprint(info)
