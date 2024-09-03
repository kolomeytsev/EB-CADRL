from simulator.utils.test_utils import configure_env_policy_robot
from simulator.utils.info import *
import numpy as np
import tempfile


MODEL_WEIGHTS_PATH = "model_weights/sarl_model_baseline.pth"
POLICY_CONFIG_PATH = "configs/test_configs/test_policy_configs/policy.config"
ENV_CONFIG_PATH = "configs/test_configs/test_env_configs/env_adults_3_bikes_3_static_10.config"
CASE_ID = 2


def run_test_save_load_map():
    env_first, _, _ = configure_env_policy_robot(
        ENV_CONFIG_PATH, POLICY_CONFIG_PATH, MODEL_WEIGHTS_PATH)
    with tempfile.NamedTemporaryFile() as tmp:
        ob, local_map = env_first.reset('test', test_case=CASE_ID, save_scene_path=tmp.name)

        env_second, _, _ = configure_env_policy_robot(
            ENV_CONFIG_PATH, POLICY_CONFIG_PATH, MODEL_WEIGHTS_PATH)
        env_second.reset('test', load_scene_path=tmp.name)

    info = {
        "info": "no_additional_info"
    }
    return np.array_equal(env_first.scene.map, env_second.scene.map), info


if __name__ == "__main__":
    res, info = run_test_save_load_map()
    if res:
        print("Test passed!")
    else:
        print("Error!")
