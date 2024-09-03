#!/bin/bash
export ENV_CONFIG="configs/env_configs/humans_8_bikes_8_child_8_static_3_35_sec_new_reward.config"
export TRAIN_CONFIG="configs/train_configs/train_50k_8x.config"
export POLICY_CONFIG="configs/policy_configs/policy_x2_no_agent_type.config"
export OUTPUT_DIR="data/humans_8_bikes_8_child_8_static_3_35_sec_new_reward_no_agent_type"
export STEP_SIZE=256

python train.py --policy sarl --env_config $ENV_CONFIG --train_config $TRAIN_CONFIG --policy_config $POLICY_CONFIG --output_dir $OUTPUT_DIR --end_iteration $STEP_SIZE
sleep 5
for resume_iteration in {256..50000..256}
do
    sum=$(( $resume_iteration + $STEP_SIZE ))
    python train.py --policy sarl --env_config $ENV_CONFIG --train_config $TRAIN_CONFIG --policy_config $POLICY_CONFIG --output_dir $OUTPUT_DIR --resume --resume_iteration $resume_iteration --end_iteration $sum
    if [ $? -ne 0 ]; then
        echo "Python script encountered an error at iteration $resume_iteration. Exiting..."
        break
    fi
    sleep 5
done
