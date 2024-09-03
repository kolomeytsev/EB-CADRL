from simulator.agents.agent import Agent
from simulator.utils.state import JointState
from simulator.utils.utils import AgentType
import numpy as np


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.action_index = None
        self.attention_weights = None
        self.last_state = None
        self.humans_in_FOV = None
        self.agent_type = AgentType.ROBOT

    def act(self, ob, local_map=None, env=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if local_map is not None:
            # action = self.policy.predict(state, local_map, self)
            action = self.policy.predict(state, env)
        else:
            action = self.policy.predict(state, env)
        return action

    def get_reward(self, local_map):
        full_state = np.expand_dims(self.last_state, axis=0)
        self_state = np.expand_dims(self.last_state[0, 0:6], axis=0)
        reward = self.policy.get_reward(full_state, self_state, local_map, self.humans_in_FOV)
        return reward
