from simulator.utils.test_utils import configure_env_policy_robot
from simulator.utils.info import *


MODEL_WEIGHTS_PATH = "model_weights/sarl_model_baseline.pth"
POLICY_CONFIG_PATH = "configs/test_configs/test_policy_configs/policy.config"


def run_scene_simulation(env_config_path, scene_path):
    env, policy, robot = configure_env_policy_robot(
        env_config_path, POLICY_CONFIG_PATH, MODEL_WEIGHTS_PATH
    )

    ob, local_map = env.reset("test", load_scene_path=scene_path)

    done = False
    while not done:
        action = robot.act(ob, local_map=local_map, env=env)
        ob, local_map_new, reward, done, info = env.step(action)

    success = isinstance(info, ReachGoal)
    result = {"info": str(info)}
    return success, result
