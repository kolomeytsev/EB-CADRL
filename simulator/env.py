import logging
import gym
import numpy as np
import math
import cv2

from simulator.utils.reward import Reward
from simulator.utils.collisions import compute_collision_agent_with_robot
from simulator.utils.render import (
    render_trajectory,
    render_am,
    render_traj_3D,
    render_og,
    render_video,
)
from simulator.scene.scene_generator import SceneGenerator


class EntityBasedCollisionAvoidance(gym.Env):
    metadata = {"render.modes": ["adult"]}
    PHASES = ["train", "val", "test"]

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be adult, bicycle, child or robot.
        adults, bicycles and children are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.name = "EntityBasedCollisionAvoidance"

        self.time_step = None
        self.robot = None
        self.other_robots = None
        self.global_time = None

        # simulation configuration
        self.case_capacity = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        # for static map
        self.submap_size_m = None
        self.local_maps = None

        # for angular map
        self.angular_map_max_range = None
        self.angular_map_dim = None
        self.angular_map_min_angle = None
        self.angular_map_max_angle = None
        self.local_maps_angular = None

        self.static_obstacles_as_pedestrians = None

    def configure(self, config):
        self.config = config

        self.scene = SceneGenerator(config)

        self.time_step = config.getfloat("env", "time_step")
        self.time_limit = config.getint("env", "time_limit")

        self.reward = Reward(config)

        self.case_capacity = {
            "train": np.iinfo(np.uint32).max - 2000,
            "val": 1000,
            "test": 1000,
        }

        self.last_circle_radius = self.scene.circle_radius

        # self.adult_policy = 'orca'

        # Static parameters
        self.use_grid_map = config.getboolean("map", "use_grid_map")

        if self.use_grid_map:
            self.submap_size_m = config.getfloat("map", "submap_size_m")
        else:
            self.angular_map_max_range = config.getfloat("map", "angular_map_max_range")
            self.angular_map_dim = config.getint("map", "angular_map_dim")
            self.angular_map_min_angle = config.getfloat("map", "angle_min") * np.pi
            self.angular_map_max_angle = config.getfloat("map", "angle_max") * np.pi

    def set_robot(self, robot):
        self.robot = robot
        self.scene.set_robot(robot)
        self.reward.set_robot(robot)

    def load_environment(self, env_config):
        raise Exception("Not yet implemented")

    def get_adult_times(self):
        """
        Run the whole simulation to the end and compute the average time for adult to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).
        :return:
        """
        raise Exception("Not yet implemented")

    def reset_times(self, phase):
        if phase == "test":
            self.adult_times = [0] * self.scene.adult_num
        else:
            self.adult_times = [0] * (
                self.scene.adult_num if self.robot.policy.multiagent_training else 1
            )

        if phase == "test":
            self.bicycle_times = [0] * self.scene.bicycle_num
        else:
            self.bicycle_times = [0] * (
                self.scene.bicycle_num if self.robot.policy.multiagent_training else 1
            )

        if phase == "test":
            self.children_times = [0] * self.scene.children_num
        else:
            self.children_times = [0] * (
                self.scene.children_num if self.robot.policy.multiagent_training else 1
            )

    def reset(
        self,
        phase="test",
        test_case=None,
        imitation_learning=False,
        compute_local_map=True,
        save_scene_path=None,
        load_scene_path=None,
        scene_number=None,
    ):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and adults
        :return:
        """
        if self.robot is None:
            raise AttributeError("robot has to be set!")
        self.phase = phase
        assert phase in self.PHASES, "phase must be one of {}".format(self.PHASES)
        if test_case is not None:
            self.scene.case_counter[phase] = test_case

        self.global_time = 0

        self.reset_times(phase)

        counter_offset = {
            "train": self.case_capacity["val"] + self.case_capacity["test"],
            "val": 0,
            # 'test': 100000,
            "test": self.case_capacity["val"],
        }
        self.robot.set(
            0, -self.scene.circle_radius, 0, self.scene.circle_radius, 0, 0, np.pi / 2
        )

        if load_scene_path is not None:
            self.scene.load_scene(phase, load_scene_path)
        else:
            # assert test_case is not None, "test_case must not be None!"
            self.scene.generate_random_scene(
                counter_offset, phase, save_scene_path, scene_number=scene_number
            )

        for agent in (
            [self.robot] + self.scene.adults + self.scene.bicycles + self.scene.children
        ):
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step
            # if agent.policy.name == 'ORCA':
            #     agent.policy.reset()

        self.states = list()
        self.local_maps = list()
        self.local_maps_angular = list()
        if hasattr(self.robot.policy, "action_values"):
            self.action_values = list()
        if hasattr(self.robot.policy, "get_attention_weights"):
            self.attention_weights = list()

        # get current observation
        ob = [
            agent.get_observable_state()
            for agent in self.scene.adults + self.scene.bicycles + self.scene.children
        ]
        if self.robot.policy.name != "SDOADRL":  # and self.robot.policy.name != 'ORCA':
            ob += self.scene.static_obstacles_as_pedestrians

        local_map = None
        if compute_local_map:
            if self.use_grid_map:
                local_map = self.get_local_map(self.robot.get_full_state())
            else:
                local_map = self.get_local_map_angular(self.robot.get_full_state())
        if self.robot.policy.name == "ORCA":
            return ob, self.scene.obstacle_vertices, local_map
        else:
            return ob, local_map
        # return ob, local_map

    def onestep_lookahead(self, action):
        next_agent_states, _, reward, done, info = self.step(action, update=False)
        return next_agent_states, reward, done, info

    def collision_detection_between_agents(self, all_agents):
        adult_num = len(all_agents)
        # Optimize
        for i in range(adult_num):
            for j in range(i + 1, adult_num):
                dx = all_agents[i].px - all_agents[j].px
                dy = all_agents[i].py - all_agents[j].py
                dist = (
                    (dx**2 + dy**2) ** (1 / 2)
                    - all_agents[i].radius
                    - all_agents[j].radius
                )
                if dist < 0:
                    # detect collision but don't take adults' collision into account
                    logging.debug("Collision happens between adults in step()")

    def compute_collision_with_obstacle(self, action, border):
        collision_obstacle = False
        px, py = self.robot.compute_position(action, self.time_step)  # next pos
        robot_idx_map_x = int(
            round((px + self.scene.map_size_m / 2.0) / self.scene.map_resolution)
        )
        robot_idx_map_y = int(
            round((py + self.scene.map_size_m / 2.0) / self.scene.map_resolution)
        )
        robot_size_map = int(
            np.ceil(self.robot.radius / np.sqrt(2.0) / self.scene.map_resolution)
        )

        start_idx_x = robot_idx_map_x - robot_size_map
        end_idx_x = start_idx_x + robot_size_map * 2

        start_idx_x = max(start_idx_x, 0)
        end_idx_x = min(
            end_idx_x, int(round(self.scene.map_size_m / self.scene.map_resolution))
        )

        start_idx_y = robot_idx_map_y - robot_size_map
        end_idx_y = start_idx_y + robot_size_map * 2

        start_idx_y = max(start_idx_y, 0)
        end_idx_y = min(
            end_idx_y, int(round(self.scene.map_size_m / self.scene.map_resolution))
        )

        if end_idx_x > start_idx_x and end_idx_y > start_idx_y:
            map_around_robot = self.scene.map[
                start_idx_x:end_idx_x, start_idx_y:end_idx_y
            ]
            if np.sum(map_around_robot) < map_around_robot.size:
                collision_obstacle = True

        # collision with border
        if border is not None:
            if (
                px <= border[0][0] + self.robot.radius
                or px >= border[0][1] - self.robot.radius
                or py <= border[1][0] + self.robot.radius
                or py >= border[1][1] - self.robot.radius
            ):
                collision_obstacle = True

        # check if robot is closer to any obstacle than the discomfort dist
        closeToObstacle = False
        robot_size_map = int(
            np.ceil(
                (self.robot.radius + self.reward.discomfort_dist)
                / self.scene.map_resolution
            )
        )

        start_idx_x = robot_idx_map_x - robot_size_map
        end_idx_x = start_idx_x + robot_size_map * 2
        start_idx_x = max(start_idx_x, 0)
        end_idx_x = min(
            end_idx_x, int(round(self.scene.map_size_m / self.scene.map_resolution))
        )
        start_idx_y = robot_idx_map_y - robot_size_map
        end_idx_y = start_idx_y + robot_size_map * 2
        start_idx_y = max(start_idx_y, 0)
        end_idx_y = min(
            end_idx_y, int(round(self.scene.map_size_m / self.scene.map_resolution))
        )
        if end_idx_x > start_idx_x and end_idx_y > start_idx_y:
            larger_map_around_robot = self.scene.map[
                start_idx_x:end_idx_x, start_idx_y:end_idx_y
            ]
            if np.sum(larger_map_around_robot) < larger_map_around_robot.size:
                closeToObstacle = True

        return collision_obstacle

    def compute_collision_with_agents(self, agents, action):
        collision_with_agent = False
        dmin = float("inf")
        for i, agent in enumerate(agents):
            dmin, new_collision = compute_collision_agent_with_robot(
                agent, self.robot, action, dmin, self.time_step
            )
            collision_with_agent |= new_collision
            if collision_with_agent:
                break
        return collision_with_agent, dmin

    def compute_collisions(self, all_agents, action, border):
        # collision detection
        collision_adult, dmin_adult = self.compute_collision_with_agents(
            self.scene.adults, action
        )
        collision_bicycle, dmin_bicycle = self.compute_collision_with_agents(
            self.scene.bicycles, action
        )
        collision_child, dmin_child = self.compute_collision_with_agents(
            self.scene.children, action
        )
        collision_obstacle = self.compute_collision_with_obstacle(action, border)

        # To long time to compute:
        # self.collision_detection_between_agents(all_agents)
        return (
            dmin_adult,
            dmin_bicycle,
            dmin_child,
            collision_adult,
            collision_bicycle,
            collision_obstacle,
            collision_child,
        )

    def compute_step_update(self, action, agents_actions, all_agents):
        """
        store state, action value and attention weights
        """
        self.states.append(
            [
                self.robot.get_full_state(),
                [adult.get_full_state() for adult in self.scene.adults],
                [bicycle.get_full_state() for bicycle in self.scene.bicycles],
                [child.get_full_state() for child in self.scene.children],
            ]
        )

        if hasattr(self.robot.policy, "action_values"):
            self.action_values.append(self.robot.policy.action_values)
        if hasattr(self.robot.policy, "get_attention_weights"):
            self.attention_weights.append(self.robot.policy.get_attention_weights())

        # update all agents
        self.robot.step(action)
        for i, adult_action in enumerate(agents_actions):
            all_agents[i].step(adult_action)

        self.global_time += self.time_step

        for i, adult in enumerate(self.scene.adults):
            # only record the first time the adult reaches the goal
            if self.adult_times[i] == 0 and adult.reached_destination():
                self.adult_times[i] = self.global_time

        for i, bicycle in enumerate(self.scene.bicycles):
            # only record the first time the bicycle reaches the goal
            if self.bicycle_times[i] == 0 and bicycle.reached_destination():
                self.bicycle_times[i] = self.global_time

        for i, child in enumerate(self.scene.children):
            # only record the first time the child reaches the goal
            if self.children_times[i] == 0 and child.reached_destination():
                self.children_times[i] = self.global_time

        # compute the observation
        if self.robot.sensor == "coordinates":
            ob = [agent.get_observable_state() for agent in all_agents]
        elif self.robot.sensor == "RGB":
            raise NotImplementedError

        return ob

    def step(self, action, update=True, compute_local_map=True, border=None):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        agents_actions = []
        all_agents = self.scene.adults + self.scene.bicycles + self.scene.children
        for agent_i, agent in enumerate(all_agents):
            # observation for adults and bicycles is always coordinates
            ob = [
                other_agent.get_observable_state()
                for other_agent in all_agents
                if other_agent != agent
            ]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]

            agents_action = agent.act(ob)  # adult.act(ob, self.scene.obstacle_vertices)
            agents_actions.append(agents_action)

            # TODO uncomment:
            """
            if self.adult_policy == 'orca':
                agents_actions.append(adult.act(ob, self.scene.obstacle_vertices))
            elif self.adult_policy == 'sdoadrl':
                agents_actions.append(
                    adult.act(
                        ob,
                        self.get_local_map_angular(
                            adult.get_full_state(),
                            append=False)))
            elif self.adult_policy == 'random':
                agents_actions.append(adult.act())
            else:
                agents_actions.append(adult.act(ob))
            """

        (
            dmin_adult,
            dmin_bicycle,
            dmin_child,
            collision_adult,
            collision_bicycle,
            collision_obstacle,
            collision_child,
        ) = self.compute_collisions(all_agents, action, border)

        reward, done, info = self.reward.compute(
            dmin_adult,
            dmin_bicycle,
            dmin_child,
            collision_adult,
            collision_bicycle,
            collision_obstacle,
            collision_child,
            action,
            self.global_time,
        )

        if update:
            ob = self.compute_step_update(action, agents_actions, all_agents)
        else:
            if self.robot.sensor == "coordinates":
                ob = [
                    agent.get_next_observable_state(action)
                    for agent, action in zip(all_agents, agents_actions)
                ]
            elif self.robot.sensor == "RGB":
                raise NotImplementedError

        if self.robot.policy.name != "SDOADRL":  # and self.robot.policy.name != 'ORCA':
            ob += self.scene.static_obstacles_as_pedestrians

        local_map = None
        if compute_local_map:
            if self.use_grid_map:
                local_map = self.get_local_map(self.robot.get_full_state())
            else:
                local_map = self.get_local_map_angular(self.robot.get_full_state())
        return ob, local_map, reward, done, info

    def calculate_angular_map_distances(
        self, vertex, edge, theta, radial_dist_vector, rad_indeces, locations
    ):
        radial_resolution = (
            self.angular_map_max_angle - self.angular_map_min_angle
        ) / float(self.angular_map_dim)
        px = (vertex[0] - edge[0]) * np.cos(theta) + (vertex[1] - edge[1]) * np.sin(
            theta
        )
        py = (vertex[1] - edge[1]) * np.cos(theta) - (vertex[0] - edge[0]) * np.sin(
            theta
        )
        phi = math.atan2(py, px)

        rad_idx = int((phi - self.angular_map_min_angle) / float(radial_resolution))

        distance = np.linalg.norm([px, py])
        if rad_idx >= 0 and rad_idx < self.angular_map_dim:
            radial_dist_vector[rad_idx] = min(radial_dist_vector[rad_idx], distance)

        for rad_idx_old, location in zip(rad_indeces, locations):
            if abs(rad_idx - rad_idx_old) > np.pi / radial_resolution:
                wrapped = True
                idx_diff = (
                    self.angular_map_dim - rad_idx + rad_idx_old
                    if rad_idx > rad_idx_old
                    else self.angular_map_dim - rad_idx_old + rad_idx
                )
            else:
                wrapped = False
                idx_diff = abs(rad_idx - rad_idx_old)
            for i in range(idx_diff):
                if (rad_idx < rad_idx_old and not wrapped) or (
                    rad_idx > rad_idx_old and wrapped
                ):
                    if (rad_idx + i) >= 0 and (rad_idx + i) < self.angular_map_dim:
                        px = (
                            vertex[0]
                            + i / float(idx_diff) * (location[0] - vertex[0])
                            - edge[0]
                        ) * np.cos(theta) + (
                            vertex[1]
                            + i / float(idx_diff) * (location[1] - vertex[1])
                            - edge[1]
                        ) * np.sin(
                            theta
                        )
                        py = (
                            vertex[1]
                            + i / float(idx_diff) * (location[1] - vertex[1])
                            - edge[1]
                        ) * np.cos(theta) - (
                            vertex[0]
                            + i / float(idx_diff) * (location[0] - vertex[0])
                            - edge[0]
                        ) * np.sin(
                            theta
                        )
                        obstacle_value_in_slice = np.linalg.norm([px, py])
                        radial_dist_vector[(rad_idx + i) % self.angular_map_dim] = min(
                            radial_dist_vector[(rad_idx + i) % self.angular_map_dim],
                            obstacle_value_in_slice,
                        )
                else:
                    if (rad_idx_old + i) >= 0 and (
                        rad_idx_old + i
                    ) < self.angular_map_dim:
                        px = (
                            location[0]
                            + i / float(idx_diff) * (vertex[0] - location[0])
                            - edge[0]
                        ) * np.cos(theta) + (
                            location[1]
                            + i / float(idx_diff) * (vertex[1] - location[1])
                            - edge[1]
                        ) * np.sin(
                            theta
                        )
                        py = (
                            location[1]
                            + i / float(idx_diff) * (vertex[1] - location[1])
                            - edge[1]
                        ) * np.cos(theta) - (
                            location[0]
                            + i / float(idx_diff) * (vertex[0] - location[0])
                            - edge[0]
                        ) * np.sin(
                            theta
                        )
                        obstacle_value_in_slice = np.linalg.norm([px, py])
                        radial_dist_vector[
                            (rad_idx_old + i) % self.angular_map_dim
                        ] = min(
                            radial_dist_vector[
                                (rad_idx_old + i) % self.angular_map_dim
                            ],
                            obstacle_value_in_slice,
                        )

        rad_indeces.append(rad_idx)
        locations.append(vertex)

    def get_local_map_angular(self, ob, normalize=True, append=True):
        """
        Compute the distance to surrounding objects in a radially discretized way.
        For each element there will be a floating point distance to the closest object in this sector.
        This allows to preserve the continuous aspect of the distance vs. a standard grid.

        !!! Attention: 0 angle is at the negative x-axis.

        number_elements: radial discretization
        relative_positions: relative positions of the surrounding objects in the local frame
        max_range: maximum range of the distance measurement

        returns:
        radial_dist_vector: contains the distance to the closest object in each sector
        """
        radial_dist_vector = self.angular_map_max_range * np.ones(
            [self.angular_map_dim]
        )
        radial_resolution = (
            self.angular_map_max_angle - self.angular_map_min_angle
        ) / float(self.angular_map_dim)
        agent_edges = []
        for s1, s2 in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            agent_edges.append((ob.px + s1 * ob.radius, ob.py + s2 * ob.radius))

        for obstacle in self.scene.obstacle_vertices:
            for edge in agent_edges:
                rad_indeces = []
                locations = []
                for vertex in obstacle:
                    self.calculate_angular_map_distances(
                        vertex,
                        edge,
                        ob.theta,
                        radial_dist_vector,
                        rad_indeces,
                        locations,
                    )

        for obstacle in self.scene.obstacle_vertices:
            for vertex in obstacle:
                rad_indeces = []
                locations = []
                for edge in agent_edges:
                    self.calculate_angular_map_distances(
                        vertex,
                        edge,
                        ob.theta,
                        radial_dist_vector,
                        rad_indeces,
                        locations,
                    )

        if normalize:
            radial_dist_vector /= float(self.angular_map_max_range)

        if append:
            self.local_maps_angular.append(radial_dist_vector)
        return radial_dist_vector

    def get_local_map(self, ob, append=True):
        """
        Extract a binary submap around the robot.
        @param ob: Full state of the robot.
        @return Binary submap rotated around robot
        """
        THRESHOLD_VALUE = 0.9
        center_idx_x = int(
            round((ob.px + self.scene.map_size_m / 2.0) / self.scene.map_resolution)
        )
        center_idx_y = int(
            round((ob.py + self.scene.map_size_m / 2.0) / self.scene.map_resolution)
        )
        size_submap = int(round(self.submap_size_m / self.scene.map_resolution))

        start_idx_x = int(round(center_idx_x - np.floor(size_submap / 2.0)))
        start_idx_y = int(round(center_idx_y - np.floor(size_submap / 2.0)))
        end_idx_x = start_idx_x + size_submap - 1
        end_idx_y = start_idx_y + size_submap - 1
        grid = np.ones((size_submap, size_submap))
        # Compute end indices (assure size of submap is correct, if out of
        # bounds)
        max_idx_x = self.scene.map.shape[0] - 1
        max_idx_y = self.scene.map.shape[1] - 1

        start_grid_x = 0
        start_grid_y = 0
        end_grid_x = size_submap - 1
        end_grid_y = size_submap - 1

        if start_idx_x < 0:
            start_grid_x = -start_idx_x
            start_idx_x = 0
        elif end_idx_x > max_idx_x:
            end_grid_x = end_grid_x - (end_idx_x - max_idx_x)
            end_idx_x = max_idx_x
        if start_idx_y < 0:
            start_grid_y = -start_idx_y
            start_idx_y = 0
        elif end_idx_y > max_idx_y:
            end_grid_y = end_grid_y - (end_idx_y - max_idx_y)
            end_idx_y = max_idx_y

        if (
            start_grid_y > end_grid_y
            or start_idx_y > end_idx_y
            or start_idx_x > end_idx_x
            or start_grid_x > end_grid_x
        ):
            grid_binary = grid
        else:
            grid[start_grid_x:end_grid_x, start_grid_y:end_grid_y] = self.scene.map[
                start_idx_x:end_idx_x, start_idx_y:end_idx_y
            ]
            grid = self.rotate_grid_around_center(
                grid, (-ob.theta + math.pi / 2) * 180 / math.pi
            )
            grid_binary = np.zeros_like(grid)
            indeces = grid > THRESHOLD_VALUE
            grid_binary[indeces] = 1
        if append:
            self.local_maps.append(grid_binary)
        return grid_binary

    def rotate_grid_around_center(self, grid, angle):
        """
        Rotate grid into direction of robot heading.
            @param grid: Grid to be rotated
            @param angle: Angle to rotate the grid by
            @return The rotated grid
        """
        grid = grid.copy()
        rows, cols = grid.shape
        M = cv2.getRotationMatrix2D(
            center=(rows / 2.0, cols / 2.0), angle=angle, scale=1
        )
        grid = cv2.warpAffine(grid, M, (rows, cols), borderValue=1)

        return grid

    def render(
        self, mode, output_file=None, deconv=None, plot_agents_goals_flag=True, frame=0
    ):
        """
        Visualizes the environment.
            @param mode: Choose 'video' for the rendering
        """
        if mode == "traj":
            render_trajectory(
                self.states,
                self.scene.adults,
                self.scene.bicycles,
                self.scene.children,
                self.robot.radius,
                self.time_step,
                self.scene.obstacle_vertices,
                self.last_circle_radius,
            )
        elif mode == "am":
            render_am(
                frame,
                self.states,
                self.scene.obstacle_vertices,
                self.local_maps_angular,
                self.angular_map_max_range,
                self.angular_map_dim,
                self.angular_map_max_angle,
                self.angular_map_min_angle,
                self.robot.radius,
            )
        elif mode == "traj3D":
            render_traj_3D(self.states, self.last_circle_radius, self.scene.adults)
        elif mode == "og":
            render_og(
                frame,
                self.states,
                self.scene.obstacle_vertices,
                self.use_grid_map,
                self.local_maps,
                self.submap_size_m,
                self.scene.map_resolution,
                self.robot.radius,
            )
        elif mode == "video":
            render_video(
                self.states,
                self.last_circle_radius,
                self.robot.radius,
                self.scene.adults,
                self.scene.bicycles,
                self.scene.children,
                self.scene.static_obstacles_as_pedestrians,
                self.local_maps,
                self.use_grid_map,
                self.scene.obstacle_vertices,
                self.robot.kinematics,
                self.scene.adult_num,
                self.submap_size_m,
                self.scene.map_resolution,
                self.time_step,
                self.angular_map_max_angle,
                self.angular_map_min_angle,
                self.angular_map_dim,
                self.angular_map_max_range,
                self.local_maps_angular,
                self.other_robots,
                self.attention_weights,
                output_file=output_file,
                deconv=deconv,
                plot_agents_goals_flag=plot_agents_goals_flag,
            )
        else:
            raise NotImplementedError
