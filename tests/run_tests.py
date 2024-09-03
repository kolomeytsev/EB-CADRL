from test_basic_simulation import run_basic_simulation
from test_basic_train import run_basic_train
from test_scene_simulation import run_scene_simulation
from test_save_load_map import run_test_save_load_map
from tests.test_collisions_simulation import run_test_collisions
import pprint


def print_test_results(results):
    print("results:")
    pprint.pprint(results)
    if all([v[0] for k, v in results.items()]):
        print("All tests successfully passed!")
    else:
        print("There are some errors in tests!")
        for k, v in results.items():
            if not v[0]:
                print(f"Test '{k}' failed! Reason: {v[1]}")


def run_tests():
    results = {}
    test_configs = [
        "configs/test_configs/test_env_configs/env_adults_3_bikes_3_static_2.config",
        "configs/test_configs/test_env_configs/env_adults_3_bikes_3.config",
        "configs/test_configs/test_env_configs/env_adults_5.config",
    ]
    for path in test_configs:
        test_name = path.split('/')[-1]
        results[test_name] = run_basic_simulation(path)

    test_configs_scenes = [
        (
            "configs/test_configs/test_env_configs/env_adults_3_bikes_3_static_10.config",
            "tests/test_scenes/test_scene_adults_3_bikes_3_static_10.json"
        ),
    ]
    for env_config_path, scene_path in test_configs_scenes:
        test_name = scene_path.split('/')[-1]
        results[test_name] = run_scene_simulation(env_config_path, scene_path)

    results["test_save_load_map"] = run_test_save_load_map()
    results["test_collisions"] = run_test_collisions()
    results["test_basic_train"] = run_basic_train()

    print_test_results(results)


if __name__ == "__main__":
    run_tests()
