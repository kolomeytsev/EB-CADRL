class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, obj_type=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

        self.obj_type = obj_type

    def __add__(self, other):
        return other + (
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        )

    def __str__(self):
        return " ".join(
            [
                str(x)
                for x in [
                    self.px,
                    self.py,
                    self.vx,
                    self.vy,
                    self.radius,
                    self.gx,
                    self.gy,
                    self.v_pref,
                    self.theta,
                    self.obj_type,
                ]
            ]
        )


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius, obj_type=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

        self.obj_type = obj_type

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.obj_type)

    def __str__(self):
        return " ".join(
            [
                str(x)
                for x in [
                    self.px,
                    self.py,
                    self.vx,
                    self.vy,
                    self.radius,
                    self.obj_type,
                ]
            ]
        )


class JointState(object):
    def __init__(self, self_state, agent_states):
        # print("!!!(comment 3 lines)")
        assert isinstance(self_state, FullState)
        for agent_state in agent_states:
            assert isinstance(agent_state, ObservableState)

        self.self_state = self_state
        self.agent_states = agent_states
