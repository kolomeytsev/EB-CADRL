import gym
from gym.envs.registration import register

register(
    id='CrowdSimStatic-v0',
    entry_point='simulator.simulator_static:CrowdSimStatic',
)
