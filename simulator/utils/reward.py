import numpy as np
from numpy.linalg import norm
from simulator.utils.action import ActionXY, ActionRot, ActionXYRot
from simulator.utils.info import *
import logging

def compute_time_reward(x, time_max, time_good):
    if x < time_good:
        return 1
    elif time_good <= x <= time_max:
        return (time_max - x) / (time_max - time_good)
    else:
        return 0


class Reward:

    def __init__(self, config):
        self.new_reward = config.getboolean('reward', 'new_reward', fallback=False)
        self.time_max = config.getfloat('reward', 'time_max', fallback=None)
        self.max_goal_distance = config.getfloat('reward', 'max_goal_distance', fallback=None)

        self.time_good = config.getfloat('reward', 'time_good', fallback=10.0)
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty_human = config.getfloat('reward', 'collision_penalty_human', fallback=None)
        self.collision_penalty_bicycle = config.getfloat('reward', 'collision_penalty_bicycle', fallback=None)
        self.collision_penalty_obstacle = config.getfloat('reward', 'collision_penalty_obstacle', fallback=None)
        self.collision_penalty_child = config.getfloat('reward', 'collision_penalty_child', fallback=None)

        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_dist_human = config.getfloat('reward', 'discomfort_dist_human',
                                                     fallback=self.discomfort_dist)
        self.discomfort_dist_bicycle = config.getfloat('reward', 'discomfort_dist_bicycle',
                                                       fallback=self.discomfort_dist)
        self.discomfort_dist_child = config.getfloat('reward', 'discomfort_dist_child',
                                                     fallback=self.discomfort_dist)

        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.discomfort_penalty_factor_human = config.getfloat('reward', 'discomfort_penalty_factor_human',
                                                               fallback=self.discomfort_penalty_factor)
        self.discomfort_penalty_factor_bicycle = config.getfloat('reward', 'discomfort_penalty_factor_bicycle',
                                                                 fallback=self.discomfort_penalty_factor)
        self.discomfort_penalty_factor_child = config.getfloat('reward', 'discomfort_penalty_factor_child',
                                                               fallback=self.discomfort_penalty_factor)

        self.rotation_penalty_factor = config.getfloat('reward', 'rotation_penalty_factor')

        self.time_step = config.getfloat('env', 'time_step')
        self.time_limit = config.getint('env', 'time_limit')

    def set_robot(self, robot):
        self.robot = robot

    def compute(self, dmin_human, dmin_bicycle, dmin_child, collision_human, collision_bicycle,
                collision_obstacle, collision_child, action, global_time):
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        dist_to_goal = norm(end_position - np.array(self.robot.get_goal_position()))
        reaching_goal = dist_to_goal < self.robot.radius
        if self.max_goal_distance is not None:
            goal_reward = 1 - dist_to_goal / self.max_goal_distance
        else:
            goal_reward = None
        reward = 0
        if self.new_reward:
            # logging.info(f"goal_reward: {goal_reward}")
            reward = goal_reward
        if global_time >= self.time_limit:
            # reward == goal_reward
            done = True
            info = Timeout(dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif collision_child:
            reward += self.collision_penalty_child
            done = True
            info = CollisionChild(dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif collision_bicycle:
            reward += self.collision_penalty_bicycle
            done = True
            info = CollisionBicycle(dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif collision_human:
            reward += self.collision_penalty_human
            done = True
            info = CollisionHuman(dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif collision_obstacle:
            reward += self.collision_penalty_obstacle
            done = True
            info = CollisionObstacle(dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        # elif collision_other_agent:
        #     reward = 0
        #     done = True
        #     info = CollisionOtherAgent()
        elif reaching_goal:
            if self.new_reward:
                time_reward = compute_time_reward(global_time, self.time_max, self.time_good)
                # Goal Proximity Reward == 1
                reward += time_reward
            else:
                reward += self.success_reward
            done = True
            info = ReachGoal(dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif dmin_child < self.discomfort_dist_child:
            reward = (dmin_child - self.discomfort_dist_child) * self.discomfort_penalty_factor_child * self.time_step
            done = False
            info = Danger(dmin_child, dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif dmin_bicycle < self.discomfort_dist_bicycle:
            reward = (dmin_bicycle - self.discomfort_dist_bicycle) * self.discomfort_penalty_factor_bicycle * self.time_step
            done = False
            info = Danger(dmin_bicycle, dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif dmin_human < self.discomfort_dist_human:
            reward = (dmin_human - self.discomfort_dist_human) * self.discomfort_penalty_factor_human * self.time_step
            done = False
            info = Danger(dmin_human, dist_to_goal, dmin_human, dmin_bicycle, dmin_child)
        elif (isinstance(action, ActionRot) or isinstance(action, ActionXYRot)) and \
                abs(action.r) > 0 and self.rotation_penalty_factor != 0:
            reward = abs(action.r) * self.rotation_penalty_factor
            done = False
            info = Nothing(dmin_human, dmin_bicycle, dmin_child)
        else:
            reward = 0
            done = False
            info = Nothing(dmin_human, dmin_bicycle, dmin_child)

        return reward, done, info
