import numpy as np
from numpy.linalg import norm
import collections
import logging
import json

from simulator.agents.agents import Human, Bicycle, BicycleRectangle, Child
from simulator.utils.state import ObservableState
from simulator.utils.utils import AgentType

MAX_ITERATIONS_TO_GENERATE_AGENT = 100000


Obstacle = collections.namedtuple(
    "Obstacle", ["location_x", "location_y", "dim", "patch"])


class SceneGenerator:

    def __init__(self, config):
        self.config = config
        self.robot = None

        self.set_agents_number(config)

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self.case_size = {
            'train': np.iinfo(np.uint32).max - 2000,
            'val': config.getint('env', 'val_size'),
            'test': config.getint('env', 'test_size')
        }

        self.train_val_sim_human = config.get('sim', 'train_val_sim_human')
        self.test_sim_human = config.get('sim', 'test_sim_human')

        self.train_val_sim_bicycle = config.get('sim', 'train_val_sim_bicycle', fallback=None)
        self.test_sim_bicycle = config.get('sim', 'test_sim_bicycle', fallback=None)

        self.train_val_sim_children = config.get('sim', 'train_val_sim_children', fallback=None)
        self.test_sim_children = config.get('sim', 'test_sim_children', fallback=None)

        self.square_width = config.getfloat('sim', 'square_width')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')

        self.circle_radius = config.getfloat('sim', 'circle_radius')
        self.last_circle_radius = self.circle_radius

        self.map_resolution = config.getfloat('map', 'map_resolution')
        self.map_size_m = config.getfloat('map', 'map_size_m')

        self.min_wall_length = config.getint('map', 'min_wall_length', fallback=2)
        self.max_wall_length = config.getint('map', 'max_wall_length', fallback=4)

        self.num_circles = config.getint('map', 'num_circles')
        self.num_walls = config.getint('map', 'num_walls')

        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')

        if config.has_option('env', 'robot_num'):
            self.other_robots_num = config.getint('env', 'robot_num')
        else:
            self.other_robots_num = 0

        # self.write_log()

    def set_agents_number(self, config):
        self.human_num = config.getint('sim', 'human_num')
        self.bicycle_num = config.getint('sim', 'bicycle_num', fallback=0)
        self.bicycle_type = config.get('sim', 'bicycle_type', fallback=None)
        self.children_num = config.getint('sim', 'children_num', fallback=0)

    def write_log(self):
        logging.info('Training human simulation: {}, test simulation: {}'
            .format(self.train_val_sim_human, self.test_sim_human))
        logging.info('Training bicycle simulation: {}, test simulation: {}'
            .format(self.train_val_sim_bicycle, self.test_sim_bicycle))
        logging.info('Training children simulation: {}, test simulation: {}'
            .format(self.train_val_sim_children, self.test_sim_children))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('human number: {}'.format(self.human_num))
        logging.info('bicycle number: {}'.format(self.bicycle_num))
        logging.info('children number: {}'.format(self.children_num))
        logging.info('robot number: {}'.format(self.other_robots_num))

    def set_robot(self, robot):
        self.robot = robot

    def generate_circle(self, circle_index, config, max_locations):
        for _ in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
            if config is not None:
                location_x = config.getfloat('x_locations_circles', str(circle_index))
                location_y = config.getfloat('y_locations_circles', str(circle_index))
                circle_radius = config.getfloat('circle_radius', str(circle_index))
            else:
                location_x = np.random.randint(-max_locations / 2.0, max_locations / 2.0)
                location_y = np.random.randint(-max_locations / 2.0, max_locations / 2.0)
                circle_radius = (np.random.random() + 0.5) * 0.7

            location_x_m = location_x * self.map_resolution
            location_y_m = location_y * self.map_resolution

            collide = False
            if norm(
                (location_x_m - self.robot.px,
                    location_y_m - self.robot.py)) < circle_radius + self.robot.radius + self.discomfort_dist or norm(
                (location_x_m - self.robot.gx,
                    location_y_m - self.robot.gy)) < circle_radius + self.robot.radius + self.discomfort_dist:
                collide = True
            if not collide:
                break

        return circle_radius, location_x, location_y, location_x_m, location_y_m

    def generate_circles(self, obstacles, num_circles, config, inflation_rate_il, max_locations, grid_size):
        for circle_index in range(num_circles):
            circle_radius, location_x, location_y, location_x_m, location_y_m = self.generate_circle(
                circle_index, config, max_locations)

            dim = (
                int(round(2 * circle_radius / self.map_resolution)),
                int(round(2 * circle_radius / self.map_resolution))
            )
            patch = np.zeros([dim[0], dim[1]])

            obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
                                      int(round(location_y + grid_size / 2.0)), dim, patch))
            circle_radius_inflated = inflation_rate_il * circle_radius
            self.obstacle_vertices.append([
                (location_x_m + circle_radius_inflated, location_y_m + circle_radius_inflated),
                (location_x_m - circle_radius_inflated, location_y_m + circle_radius_inflated),
                (location_x_m - circle_radius_inflated, location_y_m - circle_radius_inflated),
                (location_x_m + circle_radius_inflated, location_y_m - circle_radius_inflated)
            ])

    def generate_wall(self, wall_index, config, max_locations):
        for _ in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
            if config is not None:
                location_x = config.getfloat('x_locations_walls', str(wall_index))
                location_y = config.getfloat('y_locations_walls', str(wall_index))
                x_dim = config.getfloat('x_dim', str(wall_index))
                y_dim = config.getfloat('y_dim', str(wall_index))
            else:
                location_x = np.random.randint(-max_locations / 2.0, max_locations / 2.0)
                location_y = np.random.randint(-max_locations / 2.0, max_locations / 2.0)
                if np.random.random() > 0.5:
                    # horizontal
                    x_dim = np.random.randint(self.min_wall_length, self.max_wall_length + 1)
                    y_dim = 1
                else:
                    # vertical
                    y_dim = np.random.randint(self.min_wall_length, self.max_wall_length + 1)
                    x_dim = 1

            location_x_m = location_x * self.map_resolution
            location_y_m = location_y * self.map_resolution

            collide = False
            if (abs(location_x_m - self.robot.px) < x_dim / 2.0 + self.robot.radius + self.discomfort_dist \
                    and abs(location_y_m - self.robot.py) < y_dim /
                        2.0 + self.robot.radius + self.discomfort_dist) or \
                    (abs(location_x_m - self.robot.gx) < x_dim / 2.0 + self.robot.radius +
                        self.discomfort_dist and abs(location_y_m - self.robot.gy) < y_dim /
                            2.0 + self.robot.radius + self.discomfort_dist):
                collide = True
            if not collide:
                break

        return x_dim, y_dim, location_x, location_y, location_x_m, location_y_m

    def generate_walls(self, obstacles, num_walls, config, inflation_rate_il, max_locations, grid_size):
        for wall_index in range(num_walls):
            x_dim, y_dim, location_x, location_y, location_x_m, location_y_m = self.generate_wall(
                wall_index, config, max_locations)

            dim = (
                int(round(x_dim / self.map_resolution)),
                int(round(y_dim / self.map_resolution))
            )
            patch = np.zeros([dim[0], dim[1]])

            obstacles.append(Obstacle(int(round(location_x + grid_size / 2.0)),
                                      int(round(location_y + grid_size / 2.0)), dim, patch))
            x_dim_inflated = inflation_rate_il * x_dim
            y_dim_inflated = inflation_rate_il * y_dim
            self.obstacle_vertices.append([
                (location_x_m + x_dim_inflated / 2.0, location_y_m + y_dim_inflated / 2.0),
                (location_x_m - x_dim_inflated / 2.0, location_y_m + y_dim_inflated / 2.0),
                (location_x_m - x_dim_inflated / 2.0, location_y_m - y_dim_inflated / 2.0),
                (location_x_m + x_dim_inflated / 2.0, location_y_m - y_dim_inflated / 2.0)
            ])

    def generate_static_map_input(self, max_size, phase, config=None):
        """!
        Generates randomly located static obstacles (boxes and walls) in the environment.
            @param max_size: Max size in meters of the map
        """
        num_circles = 0
        num_walls = 0
        if config is not None:
            num_circles = config.getint('general', 'num_circles')
            num_walls = config.getint('general', 'num_walls')
        else:
            if self.num_circles:
                # num_circles = np.random.randint(1, self.num_circles + 1)
                num_circles = self.num_circles
            if self.num_walls:
                # num_walls = np.random.randint(1, self.num_walls + 1)
                num_walls = self.num_walls

        self.final_num_circles = num_circles
        self.final_num_walls = num_walls
        grid_size = int(round(max_size / self.map_resolution))
        self.map = np.ones((grid_size, grid_size))
        max_locations = int(round(grid_size))

        inflation_rate_il = 1
        # Todo: delete? (hack have more distance to obstacles?)
        # if phase == 'test':
        #     inflation_rate_il = 1
        # else:
        #     inflation_rate_il = 1.25

        obstacles = []
        self.obstacle_vertices = []
        self.generate_circles(obstacles, num_circles, config, inflation_rate_il, max_locations, grid_size)
        self.generate_walls(obstacles, num_walls, config, inflation_rate_il, max_locations, grid_size)
        self.place_obstacles_on_map(obstacles, grid_size)

        if self.robot.policy.name != 'SDOADRL':  # and self.robot.policy.name != 'ORCA':
            self.create_observation_from_static_obstacles(obstacles)

        self.obstacles = obstacles

    def generate_random_scene(self, counter_offset, phase, save_scene_path=None, scene_number=None):
        # if self.randomize_attributes:
        #     self.robot.sample_random_attributes()

        """
        circle_radius = self.circle_radius * \
            min(self.robot.v_pref * 5, 1) * (1 + np.random.random() * 2)
        if circle_radius > 9:
            circle_radius = 9
        self.last_circle_radius = circle_radius
        self.robot.set(
            0, -circle_radius, 0, circle_radius, 0, 0, np.pi / 2)

        if imitation_learning:
            human_num = self.human_num
        else:
            human_num = int(round(self.human_num * (0.5 + np.random.random())))
        """

        if phase in ['train', 'val']:
            human_num = self.human_num if self.robot.policy.multiagent_training else 1
            human_rule = self.train_val_sim_human

            # TODO make bicycle_num == 1 if not multiagent_training instead of 0
            bicycle_num = self.bicycle_num if self.robot.policy.multiagent_training else 0
            bicycle_rule = self.train_val_sim_bicycle

            children_num = self.children_num if self.robot.policy.multiagent_training else 1
            children_rule = self.train_val_sim_children
        else:
            human_num = self.human_num
            human_rule = self.test_sim_human

            bicycle_num = self.bicycle_num
            bicycle_rule = self.test_sim_bicycle

            children_num = self.children_num
            children_rule = self.test_sim_children

        seed = counter_offset[phase] + self.case_counter[phase]
        if scene_number is not None:
            seed = scene_number
        else:
            seed = counter_offset[phase] + self.case_counter[phase]
        print("seed:", seed)
        np.random.seed(seed)
        self.generate_random_human_position(human_num=human_num, rule=human_rule)
        self.generate_random_bicycle_position(bicycle_num=bicycle_num, rule=bicycle_rule)
        self.generate_random_children_position(children_num=children_num, rule=children_rule)
        self.generate_static_map_input(self.map_size_m, phase)

        # case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

        if save_scene_path is not None:
            self.save_scene(save_scene_path)

    def create_observation_from_static_obstacles(self, obstacles):
        self.static_obstacles_as_pedestrians = []
        for index, obstacle in enumerate(obstacles):
            if obstacle.dim[0] == obstacle.dim[1]:  # Obstacle is a square
                px = (
                    self.obstacle_vertices[index][0][0] + self.obstacle_vertices[index][2][0]) / 2.0
                py = (
                    self.obstacle_vertices[index][0][1] + self.obstacle_vertices[index][2][1]) / 2.0
                radius = (
                    self.obstacle_vertices[index][0][0] - px) * np.sqrt(2)
                self.static_obstacles_as_pedestrians.append(
                    ObservableState(px, py, 0, 0, radius, AgentType.HUMAN_STATIC))
            elif obstacle.dim[0] > obstacle.dim[1]:  # Obstacle is rectangle
                py = (
                    self.obstacle_vertices[index][0][1] + self.obstacle_vertices[index][2][1]) / 2.0
                radius = (
                    self.obstacle_vertices[index][0][1] - py) * np.sqrt(2)
                px = self.obstacle_vertices[index][1][0] + radius
                while px < self.obstacle_vertices[index][0][0]:
                    self.static_obstacles_as_pedestrians.append(
                        ObservableState(px, py, 0, 0, radius, AgentType.HUMAN_STATIC))
                    px = px + 2 * radius
            else:  # Obstacle is rectangle
                px = (
                    self.obstacle_vertices[index][0][0] + self.obstacle_vertices[index][2][0]) / 2.0
                radius = (
                    self.obstacle_vertices[index][0][0] - px) * np.sqrt(2)
                py = self.obstacle_vertices[index][2][1] + radius
                while py < self.obstacle_vertices[index][0][1]:
                    self.static_obstacles_as_pedestrians.append(
                        ObservableState(px, py, 0, 0, radius, AgentType.HUMAN_STATIC))
                    py = py + 2 * radius

    def generate_random_bicycle_position(self, bicycle_num, rule):
        self.bicycles = []
        for _ in range(bicycle_num):
            if rule == 'circle_crossing':
                bicycle = self.generate_circle_crossing_bicycle()
            elif rule == 'square_crossing':
                if self.bicycle_type == "rectangle":
                    bicycle = BicycleRectangle(self.config, 'bicycles')
                else:
                    bicycle = Bicycle(self.config, 'bicycles')
                self.generate_square_crossing_agent(bicycle, self.bicycles)
            elif rule == 'square_crossing_old':
                if self.bicycle_type == "rectangle":
                    bicycle = BicycleRectangle(self.config, 'bicycles')
                else:
                    bicycle = Bicycle(self.config, 'bicycles')
                self.generate_square_crossing_agent_old(bicycle, self.bicycles)
            else:
                raise Exception(f"Wrong rule for bicycle: {rule}")

            self.bicycles.append(bicycle)

    def generate_random_children_position(self, children_num, rule):
        self.children = []
        for _ in range(children_num):
            if rule == 'circle_crossing':
                # child = self.generate_circle_crossing_children()
                pass
            elif rule == 'square_crossing':
                child = Child(self.config, 'children')
                self.generate_square_crossing_agent(child, self.children)
            else:
                raise Exception(f"Wrong rule for children: {rule}")

            self.children.append(child)

    def generate_static_humans(self, static_human_num, width, height):
        for i in range(static_human_num):
            # randomly initialize static objects in a square of (width, height)
            if i == 0:
                human = Human(self.config, 'humans')
                human.set(-0.5, -2.5, -0.5, -2.5, 0, 0, 0)
                self.humans.append(human)
            else:
                human = Human(self.config, 'humans')
                sign = np.random.choice([1, -1], p=[0.5, 0.5])
                for _ in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
                    px = np.random.random() * width * 0.5 * sign
                    py = (np.random.random() - 0.5) * height
                    collide = False
                    for agent in [self.robot] + self.humans:
                        if norm((px - agent.px, py - agent.py)) < \
                                                human.radius + agent.radius + self.discomfort_dist:
                            collide = True
                            break

                    colide_with_goal = norm((px - self.robot.gx, py - self.robot.gy)) < \
                                            human.radius + agent.radius + self.discomfort_dist

                    if not collide and not colide_with_goal:
                        break

                human.set(px, py, px, py, 0, 0, 0)
                self.humans.append(human)

    def generate_dynamic_humans(self, dynamic_human_num):
        for i in range(dynamic_human_num):
            # the first 2 two humans will be in the circle crossing scenarios
            # the rest humans will have a random starting and end position
            if i < dynamic_human_num // 2:
                human = self.generate_circle_crossing_human()
            else:
                human = Human(self.config, 'humans')
                self.generate_square_crossing_agent(human, self.humans)
            self.humans.append(human)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num: Number of humans to add
        :param rule: square_crossing or circle_crossing
        :return:
        """
        self.humans = []
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            for i in range(human_num):
                human = Human(self.config, 'humans')
                self.generate_square_crossing_agent(human, self.humans)
                self.humans.append(human)
        elif rule == 'circle_crossing':
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {
                0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15
            }
            dynamic_human_num = {
                1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1
            }
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    for _ in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < \
                                    human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = Human(self.config, 'humans')
                        self.generate_square_crossing_agent(human, self.humans)
                    self.humans.append(human)
        elif rule == 'mixed_2':
            # mix different raining simulation with certain distribution
            static_human_num = 6
            dynamic_human_num = 6
            self.human_num = static_human_num + dynamic_human_num
            self.generate_static_humans(static_human_num, width=6, height=8)
            self.generate_dynamic_humans(dynamic_human_num)

            # static wall of humans
            add_static_wall = False
            radius = 0.15
            if add_static_wall:
                for i in range(10):
                    human = Human(self.config, 'humans')
                    human.radius = radius
                    human.set(-2 + i * 2.1 * radius, 3, -2 + i * 2.1 * radius, 3, 0, 0, 0)
                    self.humans.append(human)
        elif rule == 'mixed_12':
            static_human_num = np.random.randint(13)
            dynamic_human_num = 12 - static_human_num
            self.human_num = static_human_num + dynamic_human_num
            self.generate_static_humans(static_human_num, width=6, height=8)
            self.generate_dynamic_humans(dynamic_human_num)
        elif rule == 'mixed_20':
            static_human_num = np.random.randint(20)
            dynamic_human_num = 20 - static_human_num
            self.human_num = static_human_num + dynamic_human_num
            self.generate_static_humans(static_human_num, width=6, height=8)
            self.generate_dynamic_humans(dynamic_human_num)
        elif rule == 'one_static':
            human = Human(self.config, 'humans')
            human.set(-2, -8, -2, -8, 0, 0, 0)
            self.humans.append(human)
            human = Human(self.config, 'humans')
            human.set(-3, -8, -3, -8, 0, 0, 0)
            self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        for _ in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            # TODO: Add noise
            px_noise = 0 # (np.random.random() - 0.5) * human.v_pref
            py_noise = 0 # (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_circle_crossing_bicycle(self):
        if self.bicycle_type == "rectangle":
            bicycle = BicycleRectangle(self.config, 'bicycles')
        else:
            bicycle = Bicycle(self.config, 'bicycles')
        if self.randomize_attributes:
            bicycle.sample_random_attributes()

        for _ in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with bicycle
            # TODO: Add noise
            px_noise = 0 # (np.random.random() - 0.5) * bicycle.v_pref
            py_noise = 0 # (np.random.random() - 0.5) * bicycle.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.bicycles:
                min_dist = bicycle.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        bicycle.set(px, py, -px, -py, 0, 0, 0)
        return bicycle

    def generate_random_start_position(self):
        half_width = self.square_width / 2
        side = np.random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            position = (np.random.uniform(-half_width, half_width), half_width)
            # goal_side = np.random.choice(['bottom', 'left', 'right'])
            goal_side = 'bottom'
        elif side == 'bottom':
            position = (np.random.uniform(-half_width, half_width), -half_width)
            # goal_side = np.random.choice(['top', 'left', 'right'])
            goal_side = 'top'
        elif side == 'left':
            position = (-half_width, np.random.uniform(-half_width, half_width))
            # goal_side = np.random.choice(['top', 'bottom', 'right'])
            goal_side = 'right'
        else:  # side == 'right'
            position = (half_width, np.random.uniform(-half_width, half_width))
            # goal_side = np.random.choice(['top', 'bottom', 'left'])
            goal_side = 'left'

        return position, goal_side

    def generate_square_crossing_agent(self, agent, other_agents):
        if self.randomize_attributes:
            agent.sample_random_attributes()

        half_width = self.square_width / 2
        for index in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
            position, goal_side = self.generate_random_start_position()
            px, py = position
            # do not collide with starting position
            collide = False
            for other_agent in [self.robot] + other_agents:
                distance_to_collision = agent.radius + other_agent.radius + self.discomfort_dist
                if norm((px - other_agent.px, py - other_agent.py)) < distance_to_collision:
                    collide = True
                    break
            if collide and index != MAX_ITERATIONS_TO_GENERATE_AGENT - 1:
                continue

            if goal_side == 'top':
                goal_position = (np.random.uniform(-half_width, half_width), half_width)
            elif goal_side == 'bottom':
                goal_position = (np.random.uniform(-half_width, half_width), -half_width)
            elif goal_side == 'left':
                goal_position = (-half_width, np.random.uniform(-half_width, half_width))
            else:  # goal_side == 'right'
                goal_position = (half_width, np.random.uniform(-half_width, half_width))
            gx, gy = goal_position
            break
        agent.set(px, py, gx, gy, 0, 0, 0)

    def generate_square_crossing_agent_old(self, agent, other_agents):
        if self.randomize_attributes:
            agent.sample_random_attributes()

        sign = np.random.choice([1, -1], p=[0.5, 0.5])
        for index in range(MAX_ITERATIONS_TO_GENERATE_AGENT):
            px = np.random.random() * self.square_width * 0.5 * sign
            py = self.square_width * 0.5
            if np.random.random() > 0.5:
                px, py = py, px

            # do not collide with starting position
            collide = False
            for other_agent in [self.robot] + other_agents:
                distance_to_collision = agent.radius + other_agent.radius + self.discomfort_dist
                if norm((px - other_agent.px, py - other_agent.py)) < distance_to_collision:
                    collide = True
                    break
            if collide and index != MAX_ITERATIONS_TO_GENERATE_AGENT - 1:
                continue

            goals_dir = [(-1, 1), (1, -1), (-1, -1)]
            goal_variant = goals_dir[np.random.randint(len(goals_dir))]
            gx = px * goal_variant[0]
            gy = py * goal_variant[1]
            # do not collide with goal
            collide = False
            if index != MAX_ITERATIONS_TO_GENERATE_AGENT - 1:
                # if index is last in MAX_ITERATIONS_TO_GENERATE_AGENT we let collision with other_agent
                for other_agent in [self.robot]:
                    distance_to_collision = agent.radius + other_agent.radius + self.discomfort_dist
                    if norm((gx - other_agent.gx, gy - other_agent.gy)) < distance_to_collision:
                        collide = True
                        break
            if not collide:
                break

        agent.set(px, py, gx, gy, 0, 0, 0)

    def generate_static_map_input_from_config(self, config):
        """!
        Generates randomly located static obstacles (boxes and walls) in the environment.
            @param max_size: Max size in meters of the map
        """
        if self.robot.policy.name != 'SDOADRL' and self.robot.policy.name != 'ORCA':
            raise NotImplementedError(
                'Static map can only be created from config for SDOADRL and ORCA.')

        num_obstacles = config.getint('general', 'num_obstacles')

        grid_size = int(round(self.map_size_m / self.map_resolution))
        self.map = np.ones((grid_size, grid_size))
        self.obstacle_vertices = []
        dim = None
        for obstacle in range(num_obstacles):
            num_vertices = config.getint(
                'locations_obstacle_' + str(obstacle), 'num_vertices')
            vertex_list = list()
            for vertex in range(num_vertices):
                location_x_m = config.getfloat(
                    'locations_obstacle_' + str(obstacle), str(vertex) + '_x')
                location_y_m = config.getfloat(
                    'locations_obstacle_' + str(obstacle), str(vertex) + '_y')
                vertex_list.append((location_x_m, location_y_m))

            self.obstacle_vertices.append([vertex for vertex in vertex_list])
            pts = np.array([
                [
                    int(round(y / self.map_resolution + grid_size / 2.0)),
                    int(round(x / self.map_resolution + grid_size / 2.0))
                ] for x, y in vertex_list
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.map, [pts], 0)

    def load_scene(self, phase, load_scene_path):
        with open(load_scene_path, "r") as f:
            scene = json.load(f)

        self.human_num = len(scene["humans"])        
        self.humans = []
        for i in range(self.human_num):
            state = scene["humans"][i]
            human = Human(self.config, 'humans')
            human.set_from_state_dict(state)
            self.humans.append(human)

        self.bicycle_num = len(scene.get("bicycles", []))
        self.bicycles = []
        for i in range(self.bicycle_num):
            state = scene["bicycles"][i]
            if self.bicycle_type == "rectangle":
                bicycle = BicycleRectangle(self.config, 'bicycles')
            else:
                bicycle = Bicycle(self.config, 'bicycles')
            bicycle.set_from_state_dict(state)
            self.bicycles.append(bicycle)

        self.children_num = len(scene.get("children", []))
        self.children = []
        for i in range(self.children_num):
            state = scene["children"][i]
            child = Child(self.config, 'children')
            child.set_from_state_dict(state)
            self.children.append(child)

        self.obstacle_vertices = scene["map"]["obstacle_vertices"]
        num_circles = scene["map"]["num_circles"]
        num_walls = scene["map"]["num_walls"]
        assert len(self.obstacle_vertices) == num_circles + num_walls, \
                                    "Error: length of obstacle_vertices != num_circles + num_walls"

        grid_size = int(round(self.map_size_m / self.map_resolution))
        self.map = np.ones((grid_size, grid_size))

        obstacles = []
        for obstacle in scene["map"]["obstacles"]:
            location_x, location_y = obstacle["location"]
            dim = obstacle["dim"]
            patch = np.zeros([dim[0], dim[1]])
            obstacles.append(Obstacle(location_x, location_y, dim, patch))

        self.place_obstacles_on_map(obstacles, grid_size)

        if self.robot.policy.name != 'SDOADRL':  # and self.robot.policy.name != 'ORCA':
            self.create_observation_from_static_obstacles(obstacles)

        # case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

    def save_scene(self, path):
        result = {
            "humans": [],
            "bicycles": [],
            "children": [],
            "map": []
        }
        for human in self.humans:
            result["humans"].append(human.get_state_dict())
        for bicycle in self.bicycles:
            result["bicycles"].append(bicycle.get_state_dict())
        for child in self.children:
            result["children"].append(child.get_state_dict())

        result_map = {
            "num_circles": self.final_num_circles,
            "num_walls": self.final_num_walls,
            "obstacle_vertices": self.obstacle_vertices,
            "obstacles": [{
                "location": (obstacle[0], obstacle[1]),
                "dim": obstacle[2]
            } for obstacle in self.obstacles]
        }
        result["map"] = result_map

        with open(path, "w") as f:
            json.dump(result, f, indent=4, sort_keys=True)

    def place_obstacles_on_map(self, obstacles, grid_size):
        for obstacle in obstacles:
            if obstacle.location_x > obstacle.dim[0] / 2.0 and \
                    obstacle.location_x < grid_size - obstacle.dim[0] / 2.0 and \
                    obstacle.location_y > obstacle.dim[1] / 2.0 and \
                    obstacle.location_y < grid_size - obstacle.dim[1] / 2.0:

                start_idx_x = int(round(obstacle.location_x - obstacle.dim[0] / 2.0))
                start_idx_y = int(round(obstacle.location_y - obstacle.dim[1] / 2.0))
                self.map[start_idx_x:start_idx_x +
                         obstacle.dim[0], start_idx_y:start_idx_y +
                         obstacle.dim[1]] = np.minimum(self.map[start_idx_x:start_idx_x +
                                                                obstacle.dim[0], start_idx_y:start_idx_y +
                                                                obstacle.dim[1]], obstacle.patch)
            else:
                for idx_x in range(obstacle.dim[0]):
                    for idx_y in range(obstacle.dim[1]):
                        shifted_idx_x = idx_x - obstacle.dim[0] / 2.0
                        shifted_idx_y = idx_y - obstacle.dim[1] / 2.0
                        submap_x = int(round(obstacle.location_x + shifted_idx_x))
                        submap_y = int(round(obstacle.location_y + shifted_idx_y))
                        if submap_x > 0 and submap_x < grid_size and submap_y > 0 and submap_y < grid_size:
                            self.map[submap_x, submap_y] = obstacle.patch[idx_x, idx_y]
