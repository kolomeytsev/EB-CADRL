import numpy as np
from simulator.policy.policy import Policy
from simulator.utils.action import ActionXY


class Linear(Policy):
    def __init__(self):
        super().__init__()
        self.name = "Linear"
        self.trainable = False
        self.kinematics = "holonomic"
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state, env=None):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy - self_state.py, self_state.gx - self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionXY(vx, vy)
        return action
