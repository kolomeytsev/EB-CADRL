#!/bin/bash
export ENV_CONFIG="configs/env_configs/adults_8_bikes_8_child_8_static_3_35_sec_new_reward_fix_static.config"
export TRAIN_CONFIG="configs/train_configs/train_50k_8x.config"
export POLICY_CONFIG="configs/policy_configs/policy_x2_agent_type.config"
export OUTPUT_DIR="data/adults_8_bikes_8_child_8_static_3_35_sec_new_reward_agent_type_fix_static"
export STEP_SIZE=64

mkdir -p validation_results

for resume_iteration in {64..50000..64}
do
    sum=$(( $resume_diteration + $STEP_SIZE ))
    python test_parallel.py --policy sarl --env_config $ENV_CONFIG --policy_config $POLICY_CONFIG --model_path data/adults_8_bikes_8_child_8_static_3_35_sec_new_reward_agent_type_fix_static/rl_model_$resume_iteration.pth --csv validation_results/rl_model_$resume_iteration.csv --start 100000 --end 100500
    if [ $? -ne 0 ]; then
        echo "Python script encountered an error at iteration $resume_iteration. Exiting..."
        break
    fi
    sleep 1
done
