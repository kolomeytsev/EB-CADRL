import unittest
import configparser
import numpy as np
from simulator.agents.agents import Human
from simulator.utils.collisions import compute_collision_agent_with_robot
from simulator.utils.action import ActionXY
from simulator.agents.robot import Robot

ENV_CONFIG_PATH = "configs/env_configs/env_humans_5_bikes_5_static_5.config"


class TestCollisionAgentWithRobot(unittest.TestCase):
    def test_no_collision_1(self):
        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)

        robot = Robot(env_config, 'robot')
        robot.kinematics = "holonomic"
        robot.set(0, 0, 0, 0, 0, 0, np.pi / 2)
        robot.radius = 1
        human = Human(env_config, 'humans')
        human.set(0, -2, 0, -2, 0, 0, 0)
        human.radius = 0.9

        time_step = 0.07
        robot.time_step = time_step
        dmin = float('inf')
        action = ActionXY(-1, -1)

        result = compute_collision_agent_with_robot(human, robot, action, dmin, time_step)
        self.assertEqual(result[1], False)

    def test_collision_1(self):
        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)

        robot = Robot(env_config, 'robot')
        robot.kinematics = "holonomic"
        robot.set(0, 0, 0, 0, 0, 0, np.pi / 2)
        robot.radius = 1
        human = Human(env_config, 'humans')
        human.set(0, -2, 0, -2, 0, 0, 0)
        human.radius = 0.9

        time_step = 0.12
        robot.time_step = time_step
        dmin = float('inf')
        action = ActionXY(-1, -1)

        result = compute_collision_agent_with_robot(human, robot, action, dmin, time_step)
        self.assertEqual(result[1], True)

    def test_no_collision_2(self):
        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)

        robot = Robot(env_config, 'robot')
        robot.kinematics = "holonomic"
        robot.set(0, 0, 0, 0, 0, 0, np.pi / 2)
        robot.radius = 1
        human = Human(env_config, 'humans')
        human.set(0, -2, 0, -2, 0, 0, 0)
        human.radius = 0.9

        time_step = 1
        robot.time_step = time_step
        dmin = float('inf')
        action = ActionXY(1, 1)

        result = compute_collision_agent_with_robot(human, robot, action, dmin, time_step)
        self.assertEqual(result[1], False)

    def test_no_collision_3(self):
        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)

        robot = Robot(env_config, 'robot')
        robot.kinematics = "holonomic"
        robot.set(0, 0, 0, 0, 0, 0, np.pi / 2)
        robot.radius = 1
        human = Human(env_config, 'humans')
        human.set(1, -2, 1, -2, 0, 0, 0)
        human.radius = 1

        time_step = 0.17
        robot.time_step = time_step
        dmin = float('inf')
        action = ActionXY(1, -1)

        result = compute_collision_agent_with_robot(human, robot, action, dmin, time_step)
        self.assertEqual(result[1], False)

    def test_collision_2(self):
        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)

        robot = Robot(env_config, 'robot')
        robot.kinematics = "holonomic"
        robot.set(0, 0, 0, 0, 0, 0, np.pi / 2)
        robot.radius = 1
        human = Human(env_config, 'humans')
        human.set(1, -2, 1, -2, 0, 0, 0)
        human.radius = 1

        time_step = 0.18
        robot.time_step = time_step
        dmin = float('inf')
        action = ActionXY(1, -1)

        result = compute_collision_agent_with_robot(human, robot, action, dmin, time_step)
        self.assertEqual(result[1], True)

    def test_collision_3(self):
        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)

        robot = Robot(env_config, 'robot')
        robot.kinematics = "holonomic"
        robot.set(1, 4, 0, 0, 0, 0, np.pi / 2)
        robot.radius = 1
        human = Human(env_config, 'humans')
        human.set(3, 5, 0, 0, 0, 0, 0)
        human.radius = 1.2

        time_step = 1.178
        robot.time_step = time_step
        dmin = float('inf')
        action = ActionXY(1, -1)

        result = compute_collision_agent_with_robot(human, robot, action, dmin, time_step)
        self.assertEqual(result[1], True)

if __name__ == '__main__':
    unittest.main()
