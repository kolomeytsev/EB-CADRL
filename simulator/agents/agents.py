from simulator.agents.agent import Agent
from simulator.agents.agent_rectangle import AgentRectangle
from simulator.utils.state import JointState
from simulator.utils.utils import AgentType
import numpy as np


class Adult(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.agent_type = AgentType.ADULT

    def act(self, ob=None, global_map=None, local_map=None):
        """
        The state for adult is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        elif local_map is not None:
            # action = self.policy.predict(state, local_map, self)
            action = self.policy.predict(state)
        else:
            action = self.policy.predict(state)

        return action


class Bicycle(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.agent_type = AgentType.BICYCLE

    def act(self, ob=None, global_map=None, local_map=None):
        """
        The state for bicycle is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        elif local_map is not None:
            # action = self.policy.predict(state, local_map, self)
            action = self.policy.predict(state)
        else:
            action = self.policy.predict(state)

        return action


class BicycleRectangle(AgentRectangle):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.agent_type = AgentType.BICYCLE

    def act(self, ob=None, global_map=None, local_map=None):
        """
        The state for bicycle is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        elif local_map is not None:
            # action = self.policy.predict(state, local_map, self)
            action = self.policy.predict(state)
        else:
            action = self.policy.predict(state)

        return action


class Child(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.agent_type = AgentType.CHILD

    def act(self, ob=None, global_map=None, local_map=None):
        """
        The state for bicycle is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if ob is None:
            return self.policy.predict(self)
        state = JointState(self.get_full_state(), ob)
        if global_map is not None:
            action = self.policy.predict(state, global_map, self)
        elif local_map is not None:
            # action = self.policy.predict(state, local_map, self)
            action = self.policy.predict(state)
        else:
            action = self.policy.predict(state)

        return action
