import numpy as np
from numpy.linalg import norm
import abc
import logging
from simulator.policy.policy_factory import policy_factory
from simulator.utils.action import ActionXY, ActionRot, ActionXYRot
from simulator.utils.state import ObservableState, FullState
from simulator.utils.utils import AgentType


class AgentRectangle(object):
    def __init__(self, config, section):
        """
        Base class for rectangle agents. Have the physical attributes of an agent.
        """
        self.visible = config.getboolean(section, "visible")
        self.v_pref = config.getfloat(section, "v_pref")

        self.radius = config.getfloat(section, "radius")

        self.width = config.getfloat(section, "width")
        self.length = config.getfloat(section, "length")

        self.half_width = self.width / 2
        self.half_length = self.length / 2

        print("self.half_width:", self.half_width)
        print("self.half_length:", self.half_length)

        self.policy = policy_factory[config.get(section, "policy")]()
        self.sensor = config.get(section, "sensor")
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None
        self.agent_type = None

    def print_info(self):
        logging.info(
            "Agent is {} and has {} kinematic constraint".format(
                "visible" if self.visible else "invisible", self.kinematics
            )
        )

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 2.0)
        self.radius = np.random.uniform(0.2, 0.5)

    def set(
        self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None, agent_type=None
    ):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
        self.agent_type = agent_type

    def get_observable_state(self):
        return ObservableState(
            self.px, self.py, self.vx, self.vy, self.radius, self.agent_type
        )

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == "holonomic":
            print("get_next_observable_state holonomic")
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(
            next_px, next_py, next_vx, next_vy, self.radius, self.agent_type
        )

    def get_full_state(self):
        return FullState(
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
            self.agent_type,
        )

    def get_state_dict(self):
        return {
            "pos": (self.px, self.py),
            "vel": (self.vx, self.vy),
            "radius": self.radius,
            "goal": (self.gx, self.gy),
            "v_pref": self.v_pref,
            "theta": self.theta,
            "agent_type:": self.agent_type,
        }

    def set_from_state_dict(self, state):
        self.px = state["pos"][0]
        self.py = state["pos"][1]
        self.vx = state["vel"][0]
        self.vy = state["vel"][1]
        self.radius = state["radius"]
        self.gx = state["goal"][0]
        self.gy = state["goal"][1]
        self.v_pref = state["v_pref"]
        self.theta = state["theta"]
        if state.get("agent_type") is not None:
            self.agent_type = AgentType(state["agent_type"])

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy
        """
        return

    def check_validity(self, action):
        if self.kinematics == "holonomic":
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot) or isinstance(action, ActionXYRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == "holonomic":
            print("compute_position holonomic")
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            if isinstance(action, ActionRot):
                px = self.px + np.cos(theta) * action.v * delta_t
                py = self.py + np.sin(theta) * action.v * delta_t
            elif isinstance(action, ActionXYRot):
                px = (
                    self.px
                    + np.cos(theta) * action.vx * delta_t
                    - np.sin(theta) * action.vy * delta_t
                )
                py = (
                    self.py
                    + np.sin(theta) * action.vx * delta_t
                    + np.cos(theta) * action.vy * delta_t
                )
            else:
                raise Exception("Wrong action type")

        return px, py

    def compute_velocity(self, action):
        self.check_validity(action)
        theta = self.theta + action.r
        if isinstance(action, ActionRot):
            vx = action.v * np.cos(theta)
            vy = action.v * np.sin(theta)
        elif isinstance(action, ActionXYRot):
            vx = action.vx * np.cos(theta) - action.vy * np.sin(theta)
            vy = action.vx * np.sin(theta) + action.vy * np.cos(theta)

        return vx, vy

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos

        if self.kinematics == "holonomic":
            print("step holonomic")
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)
            if isinstance(action, ActionRot):
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
            elif isinstance(action, ActionXYRot):
                self.vx = action.vx * np.cos(self.theta) - action.vy * np.sin(
                    self.theta
                )
                self.vy = action.vx * np.sin(self.theta) + action.vy * np.cos(
                    self.theta
                )
            else:
                raise Exception("Wrong action type")

    def reached_destination(self):
        return (
            norm(np.array(self.get_position()) - np.array(self.get_goal_position()))
            < self.radius
        )
