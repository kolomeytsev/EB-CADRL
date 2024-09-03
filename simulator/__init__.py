import gym
from gym.envs.registration import register

register(
    id='EntityBasedCollisionAvoidance-v0',
    entry_point='simulator.env:EntityBasedCollisionAvoidance',
)
