from simulator.policy.linear import Linear
from simulator.policy.orca import ORCA
from simulator.policy.orca_obstacles import ORCAObstacles


def none_policy():
    return None


policy_factory = dict()
policy_factory["linear"] = Linear
policy_factory["orca"] = ORCA
policy_factory["orca_obstacles"] = ORCAObstacles
policy_factory["none"] = none_policy
