#!/bin/bash
export ENV_CONFIG="configs/env_configs/adults_8_bikes_8_child_8_static_3_35_sec_new_reward_ablation.config"
export POLICY_CONFIG="configs/policy_configs/policy_x2_no_agent_type.config"

python rl/test_parallel.py --policy sarl --env_config $ENV_CONFIG --policy_config $POLICY_CONFIG --model_path data/sarl-gp/rl_model_val.pth --csv data/sarl-gp/test.csv --start 1000000 --end 1000009
