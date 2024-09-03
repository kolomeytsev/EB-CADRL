from simulator.utils.test_utils import configure_env_policy_robot
from simulator.utils.info import *
import pprint


POLICY_CONFIG_PATH = "configs/test_configs/test_policy_configs/policy.config"
PATH_PART = "tests/test_scenes/test_collisions/"

ENV_CONFIG_PATH_1 = "configs/test_configs/test_env_configs/env_adults_5_bikes_5_static_5.config"
SCENE_RESULTS_1 = [
    ("collision_with_adult.json", CollisionAdult),
    ("collision_with_bicycle.json", CollisionBicycle),
    ("collision_with_static.json", CollisionObstacle),
    ("no_collisions.json", ReachGoal)
]

ENV_CONFIG_PATH_2 = "configs/test_configs/test_env_configs/env_adults_5_bikes_0_static_5.config"
SCENE_RESULTS_2 = [
    ("bikes_0_collision_with_adult_1.json", CollisionAdult),
    ("bikes_0_collision_with_adult_2.json", CollisionAdult),
    ("bikes_0_no_collisions.json", ReachGoal)
]


ENV_CONFIG_PATH_3 = "configs/test_configs/test_env_configs/env_adults_5_child_5_static_5.config"
SCENE_RESULTS_3 = [
    ("collision_with_child.json", CollisionChild)
]


ALL_TESTS = [
    (ENV_CONFIG_PATH_1, SCENE_RESULTS_1),
    (ENV_CONFIG_PATH_2, SCENE_RESULTS_2),
    (ENV_CONFIG_PATH_3, SCENE_RESULTS_3)
]


def run_simulation(env, robot, scene_path):
    ob, local_map = env.reset('test', load_scene_path=scene_path)
    done = False
    while not done:
        action = robot.act(ob, local_map=local_map, env=env)
        ob, local_map_new, reward, done, info = env.step(action)
    
    return info


def run_test_collisions_with_configs(results, env_config, scene_results):
    env, policy, robot = configure_env_policy_robot(env_config, POLICY_CONFIG_PATH, policy="linear")
    for scene_path, info_result in scene_results:
        info = run_simulation(env, robot, PATH_PART + scene_path)
        success = isinstance(info, info_result)
        results.append((success, (str(info), info_result)))


def run_test_collisions():
    results = []
    for env_config, scene_results in ALL_TESTS:
        run_test_collisions_with_configs(results, env_config, scene_results)

    success = all([x[0] for x in results])
    result = {
        "info": results
    }
    return success, result


if __name__ == "__main__":
    res, info = run_test_collisions()
    if res:
        print("Test passed!")
    else:
        print("Error!")
    pprint.pprint(info)
