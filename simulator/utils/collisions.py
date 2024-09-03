import numpy as np


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def compute_collision_agent_with_robot(agent, robot, action, dmin, time_step):
    """
    Make agent the center of coordinates: (0, 0)
    shift robot by action from (px, py) to (ex, ey)
    compute distance from point (0, 0) to segment (px, py) - (ex, ey)
    """
    px = agent.px - robot.px
    py = agent.py - robot.py
    if robot.kinematics == 'holonomic':
        vx = agent.vx - action.vx
        vy = agent.vy - action.vy
    else:
        vx = agent.vx - action.v * np.cos(action.r + robot.theta)
        vy = agent.vy - action.v * np.sin(action.r + robot.theta)
    ex = px + vx * time_step
    ey = py + vy * time_step

    # closest distance between boundaries of two agents
    closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - agent.radius - robot.radius

    collision = False
    if closest_dist < 0:
        collision = True
        # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
    elif closest_dist < dmin:
        dmin = closest_dist
    return dmin, collision
