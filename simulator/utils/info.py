class Info(object):
    def __init__(self):
        self.dist_to_goal = None
        self.dmin_adult = None
        self.dmin_bicycle = None
        self.dmin_child = None

    def __str__(self):
        pass


class Timeout(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "Timeout"


class ReachGoal(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "Reaching goal"


class Danger(Info):
    def __init__(
        self,
        min_dist,
        dist_to_goal=None,
        dmin_adult=None,
        dmin_bicycle=None,
        dmin_child=None,
    ):
        super().__init__()
        self.min_dist = min_dist
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "Too close"


class CollisionAdult(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "CollisionAdult"


class CollisionBicycle(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "CollisionBicycle"


class CollisionObstacle(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "CollisionObstacle"


class CollisionChild(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "CollisionChild"


class Collision(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "Collision"


class CollisionOtherAgent(Info):
    def __init__(
        self, dist_to_goal=None, dmin_adult=None, dmin_bicycle=None, dmin_child=None
    ):
        super().__init__()
        self.dist_to_goal = dist_to_goal
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return "Collision from other agent"


class Nothing(Info):
    def __init__(self, dmin_adult=None, dmin_bicycle=None, dmin_child=None):
        super().__init__()
        self.dmin_adult = dmin_adult
        self.dmin_bicycle = dmin_bicycle
        self.dmin_child = dmin_child

    def __str__(self):
        return ""
