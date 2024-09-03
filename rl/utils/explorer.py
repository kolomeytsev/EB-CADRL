import logging
import copy
import torch
from typing import Dict
from simulator.utils.info import (ReachGoal, Collision, CollisionChild, CollisionAdult,
                                       CollisionBicycle, CollisionObstacle, Timeout, Danger)


class Explorer(object):
    PHASES = ['val', 'test']
    ORCA_POLICY = 'ORCA'

    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def run_one_episode(self, phase, update_memory, imitation_learning, episode_number):
        if self.robot.policy.name == self.ORCA_POLICY:
            ob, global_map, local_map = self.env.reset(phase)  # TODO: use global_map
        else:
            ob, local_map = self.env.reset(phase)

        done = False
        states, actions, rewards = [], [], []
        while not done:
            action = self.robot.act(ob, local_map=local_map, env=self.env)
            ob, local_map, reward, done, info = self.env.step(action)
            states.append(self.robot.policy.last_state)
            actions.append(action)
            rewards.append(reward)

            if isinstance(info, Danger):
                self.too_close += 1
                self.min_dist.append(info.min_dist)

        if isinstance(info, ReachGoal):
            self.success += 1
            self.success_times.append(self.env.global_time)
        elif isinstance(info, Collision):
            self.collision += 1
            self.collision_cases.append(episode_number)
            self.collision_times.append(self.env.global_time)
        elif isinstance(info, CollisionChild):
            self.collision_child += 1
            self.collision_cases_child.append(episode_number)
            self.collision_times_child.append(self.env.global_time)
        elif isinstance(info, CollisionAdult):
            self.collision_adult += 1
            self.collision_cases_adult.append(episode_number)
            self.collision_times_adult.append(self.env.global_time)
        elif isinstance(info, CollisionBicycle):
            self.collision_bicycle += 1
            self.collision_cases_bicycle.append(episode_number)
            self.collision_times_bicycle.append(self.env.global_time)
        elif isinstance(info, CollisionObstacle):
            self.collision_obstacle += 1
            self.collision_cases_obstacle.append(episode_number)
            self.collision_times_obstacle.append(self.env.global_time)
        elif isinstance(info, Timeout):
            self.timeout += 1
            self.timeout_cases.append(episode_number)
            self.timeout_times.append(self.env.time_limit)
        else:
            raise ValueError('Invalid end signal from environment')

        if update_memory:
            collision_flag = isinstance(info, Collision) or isinstance(info, CollisionAdult) or \
                isinstance(info, CollisionBicycle) or isinstance(info, CollisionObstacle) or \
                isinstance(info, CollisionChild)
            if isinstance(info, ReachGoal) or collision_flag:
                # only add positive(success) or negative(collision) experience in experience set
                self.update_memory(states, actions, rewards, imitation_learning)

        self.cumulative_rewards.append(self.calculate_cumulative_reward(rewards))

    def init_metrics(self):
        self.success_times = []

        self.collision_times = []
        self.collision_times_adult = []
        self.collision_times_bicycle = []
        self.collision_times_obstacle = []
        self.collision_times_child = []

        self.timeout_times = []
        self.success = 0

        self.collision = 0
        self.collision_adult = 0
        self.collision_bicycle = 0
        self.collision_child = 0
        self.collision_obstacle = 0

        self.timeout = 0
        self.too_close = 0
        self.min_dist = []
        self.cumulative_rewards = []

        self.collision_cases = []
        self.collision_cases_adult = []
        self.collision_cases_bicycle = []
        self.collision_cases_obstacle = []
        self.collision_cases_child = []

        self.timeout_cases = []

    # @profile
    def run_k_episodes(self, num_episodes, phase, update_memory=False, imitation_learning=False,
                       episode=None, print_failure=False, return_metrics=False):
        self.robot.policy.set_phase(phase)

        self.init_metrics()
        for episode_number in range(num_episodes):
            self.run_one_episode(phase, update_memory, imitation_learning, episode_number)

        self.log_results(print_failure, num_episodes, phase, episode)

        if return_metrics:
            return self.compile_metrics()

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different adult_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     adult_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     adult_num, feature_size = state.size()
            # if adult_num != 5:
            #     padding = torch.zeros((5 - adult_num, feature_size))
            #     state = torch.cat([state, padding])
            # delete far agents instead of padding
            self.memory.push((state, value))

    def calculate_cumulative_reward(self, rewards) -> float:
        return sum([
            pow(self.gamma, t * self.robot.time_step * self.robot.v_pref) * reward for t, reward in enumerate(rewards)])

    @staticmethod
    def average(input_list) -> float:
        return sum(input_list) / len(input_list) if input_list else 0

    def compile_metrics(self) -> Dict:
        return {
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "collision_rate_adult": self.collision_rate_adult,
            "collision_rate_bicycle": self.collision_rate_bicycle,
            "collision_rate_child": self.collision_rate_child,
            "collision_rate_obstacle": self.collision_rate_obstacle,
            "success": self.success,
            "collision": self.collision,
            "timeout": self.timeout,
            "avg_nav_time": self.avg_nav_time,
            "total_reward:": self.average(self.cumulative_rewards),
            "Frequency of being in danger": self.too_close / self.num_step if self.num_step else None,
            "average min separate distance in danger": self.average(self.min_dist),
            "Collision cases:": [str(x) for x in self.collision_cases],
            "Collision Adult cases:": [str(x) for x in self.collision_cases_adult],
            "Collision Bicycle cases:": [str(x) for x in self.collision_cases_bicycle],
            "Collision Child cases:": [str(x) for x in self.collision_cases_child],
            "Collision Obstacle cases:": [str(x) for x in self.collision_cases_obstacle],
            "Timeout cases": [str(x) for x in self.timeout_cases]
        }

    def log_results(self, print_failure: bool = False, num_episodes: int = None, phase: str = None,
                    episode: int = None):
        self.success_rate = self.success / num_episodes

        self.collision_rate = self.collision / num_episodes
        self.collision_rate_adult = self.collision_adult / num_episodes
        self.collision_rate_bicycle = self.collision_bicycle / num_episodes
        self.collision_rate_child = self.collision_child / num_episodes
        self.collision_rate_obstacle = self.collision_obstacle / num_episodes

        assert (self.success + self.collision + self.collision_adult + self.collision_bicycle + \
                self.collision_obstacle + self.collision_child + self.timeout) == num_episodes

        self.avg_nav_time = sum(self.success_times) / len(self.success_times) if self.success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)

        logging.info('{:<5} {}has success rate: {:.2f}, self.collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, self.success_rate, self.collision_rate, self.avg_nav_time,
                   self.average(self.cumulative_rewards)))

        logging.info('{:<5} {}self.collision rate adult: {:.2f}, self.collision rate bicycle: {:.2f},'
                     'self.collision rate child: {:.2f}, self.collision rate obstacle: {:.4f}'.
            format(phase.upper(), extra_info, self.collision_rate_adult, self.collision_rate_bicycle,
                   self.collision_rate_child, self.collision_rate_obstacle))

        self.num_step = None
        if phase in self.PHASES:
            self.num_step = sum(
                self.success_times + self.collision_times + self.collision_times_adult + \
                self.collision_times_bicycle + self.collision_times_child + \
                self.collision_times_obstacle + self.timeout_times) / self.robot.time_step

            logging.info(
                'Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                self.too_close / self.num_step,
                self.average(self.min_dist)
            )

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in self.collision_cases]))
            logging.info('Collision Adult cases: ' + ' '.join([str(x) for x in self.collision_cases_adult]))
            logging.info('Collision Bicycle cases: ' + ' '.join([str(x) for x in self.collision_cases_bicycle]))
            logging.info('Collision Child cases: ' + ' '.join([str(x) for x in self.collision_cases_child]))
            logging.info('Collision Obstacle cases: ' + ' '.join([str(x) for x in self.collision_cases_obstacle]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in self.timeout_cases]))
