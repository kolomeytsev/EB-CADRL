import logging
import copy
import torch
import time
import gym

from rl.utils.utils import configure_policy, configure_environment_and_robot

from simulator.agents.robot import Robot
import multiprocessing

# from multiprocessing import Pool, cpu_count
from simulator.utils.info import (
    ReachGoal,
    Collision,
    CollisionChild,
    CollisionAdult,
    CollisionBicycle,
    CollisionObstacle,
    Timeout,
    Danger,
)

ORCA_POLICY = "ORCA"


def average(input_list) -> float:
    return sum(input_list) / len(input_list) if input_list else 0


def calculate_cumulative_reward(rewards, gamma, time_step, v_pref) -> float:
    return sum(
        [
            pow(gamma, t * time_step * v_pref) * reward
            for t, reward in enumerate(rewards)
        ]
    )


def run_one_episode(args, phase, policy, imitation_learning, episode):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        robot, env = configure_environment_and_robot(
            args, "EntityBasedCollisionAvoidance-v0"
        )
        robot.set_policy(policy)
        robot.policy.set_phase(phase)

        if robot.policy.name == ORCA_POLICY:
            ob, global_map, local_map = env.reset(phase, episode, scene_number=episode)
        else:
            ob, local_map = env.reset(phase, scene_number=episode)

        done = False
        states, actions, rewards = [], [], []
        collision_flag = False
        torch.set_num_threads(1)
        while not done:
            action = robot.act(ob, local_map=local_map, env=env)
            ob, local_map, reward, done, info = env.step(action)
            states.append(robot.policy.last_state)
            actions.append(action)
            rewards.append(reward)

        # In a real scenario, you would likely return states, actions, rewards, and other relevant info
        collision_flag = (
            isinstance(info, Collision)
            or isinstance(info, CollisionAdult)
            or isinstance(info, CollisionBicycle)
            or isinstance(info, CollisionObstacle)
            or isinstance(info, CollisionChild)
        )

        success = 0
        collision = (
            collision_child
        ) = collision_adult = collision_bicycle = collision_obstacle = 0
        timeout = 0
        if isinstance(info, ReachGoal):
            success += 1
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, CollisionChild):
            collision_child += 1
        elif isinstance(info, CollisionAdult):
            collision_adult += 1
        elif isinstance(info, CollisionBicycle):
            collision_bicycle += 1
        elif isinstance(info, CollisionObstacle):
            collision_obstacle += 1
        elif isinstance(info, Timeout):
            timeout += 1
        else:
            raise ValueError("Invalid end signal from environment")
        success_rate = success
        collision_rate = collision
        collision_rate_adult = collision_adult
        collision_rate_bicycle = collision_bicycle
        collision_rate_child = collision_child
        collision_rate_obstacle = collision_obstacle

        assert (
            success
            + collision
            + collision_adult
            + collision_bicycle
            + collision_obstacle
            + collision_child
            + timeout
        ) == 1

        avg_nav_time = env.global_time

        extra_info = "" if episode is None else "in episode {} ".format(episode)

        cum_reward = calculate_cumulative_reward(
            rewards, policy.gamma, env.time_step, robot.v_pref
        )
        logging.info(
            "{:<5} {} 1 has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}".format(
                phase.upper(),
                extra_info,
                success_rate,
                collision_rate,
                avg_nav_time,
                cum_reward,
            )
        )

        logging.info(
            "{:<5} {}collision rate adult: {:.2f}, collision rate bicycle: {:.2f},"
            "collision rate child: {:.2f}, collision rate obstacle: {:.4f}".format(
                phase.upper(),
                extra_info,
                collision_rate_adult,
                collision_rate_bicycle,
                collision_rate_child,
                collision_rate_obstacle,
            )
        )
        return collision_flag, states, actions, rewards, info, avg_nav_time, cum_reward
    except RuntimeError as e:
        print(f"Caught RuntimeError in thread: {e}")
        logging.info("Caught RuntimeError in thread: {}".format(str(e)))
        return False, [], [], [], None, None, 0


class ParallelExplorer(object):
    PHASES = ["val", "test"]
    ORCA_POLICY = "ORCA"

    def __init__(
        self,
        env,
        robot,
        device,
        processes_num,
        memory=None,
        gamma=None,
        target_policy=None,
    ):
        self.time_step = env.time_step
        self.v_pref = robot.v_pref
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.processes_num = processes_num

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model).to(self.device)

    def process_results(self, results, update_memory, imitation_learning):
        # Example of processing results:
        for (
            collision_flag,
            states,
            actions,
            rewards,
            info,
            avg_nav_time,
            cum_reward,
        ) in results:
            if update_memory:
                # collision_flag = isinstance(info, Collision) or isinstance(info, CollisionAdult) or \
                #     isinstance(info, CollisionBicycle) or isinstance(info, CollisionObstacle) or \
                #     isinstance(info, CollisionChild)
                # if isinstance(info, ReachGoal) or collision_flag:
                #     # only add positive(success) or negative(collision) experience in experience set
                #     self.update_memory(states, actions, rewards, imitation_learning)
                self.update_memory(states, actions, rewards, imitation_learning)

    def log_val_aggregated(self, results):
        episodes_num = len(results)
        success = 0
        collision = (
            collision_child
        ) = collision_adult = collision_bicycle = collision_obstacle = 0
        timeout = 0
        cum_reward_sum = 0
        nav_time_sum = 0
        for (
            collision_flag,
            states,
            actions,
            rewards,
            info,
            avg_nav_time,
            cum_reward,
        ) in results:
            if info is None:
                continue
            cum_reward_sum += cum_reward
            nav_time_sum += avg_nav_time
            if isinstance(info, ReachGoal):
                success += 1
            elif isinstance(info, Collision):
                collision += 1
            elif isinstance(info, CollisionChild):
                collision_child += 1
            elif isinstance(info, CollisionAdult):
                collision_adult += 1
            elif isinstance(info, CollisionBicycle):
                collision_bicycle += 1
            elif isinstance(info, CollisionObstacle):
                collision_obstacle += 1
            elif isinstance(info, Timeout):
                timeout += 1
            else:
                raise ValueError("Invalid end signal from environment")

        success_rate = success / episodes_num
        collision_rate = collision / episodes_num
        collision_rate_adult = collision_adult / episodes_num
        collision_rate_bicycle = collision_bicycle / episodes_num
        collision_rate_child = collision_child / episodes_num
        collision_rate_obstacle = collision_obstacle / episodes_num
        cum_reward_avg = cum_reward_sum / episodes_num
        nav_time_avg = nav_time_sum / episodes_num

        logging.info(
            "VAL_AGGREGATED has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}".format(
                success_rate, collision_rate, nav_time_avg, cum_reward_avg
            )
        )

        logging.info(
            "VAL_AGGREGATED collision rate adult: {:.2f}, collision rate bicycle: {:.2f},"
            "collision rate child: {:.2f}, collision rate obstacle: {:.4f}".format(
                collision_rate_adult,
                collision_rate_bicycle,
                collision_rate_child,
                collision_rate_obstacle,
            )
        )

    def run_k_episodes_parallel(
        self,
        args,
        start_episode,
        end_episode,
        phase,
        update_memory=False,
        imitation_learning=False,
    ):
        torch.multiprocessing.set_sharing_strategy("file_system")
        episodes = list(range(start_episode, end_episode))
        print("run_k_episodes, num_episodes: ", len(episodes), episodes)
        t0 = time.time()
        args_list = [
            (args, phase, self.target_policy, imitation_learning, ep) for ep in episodes
        ]

        with multiprocessing.Pool(processes=self.processes_num) as pool:
            results = pool.starmap(run_one_episode, args_list)

        # Note: Properly handling model updates with multiprocessing requires careful design,
        # especially if using PyTorch and CUDA. Consider using shared_memory or moving model
        # updates outside of the multiprocessing part.
        if phase == "val":
            self.log_val_aggregated(results)
        self.process_results(results, update_memory, imitation_learning)
        print("time run_k_episodes: ", time.time() - t0)

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError("Memory or gamma value is not set!")

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.time_step * self.v_pref)
                value = sum(
                    [
                        pow(self.gamma, max(t - i, 0) * self.time_step * self.v_pref)
                        * reward
                        * (1 if t >= i else 0)
                        for t, reward in enumerate(rewards)
                    ]
                )
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
                    value = (
                        reward
                        + gamma_bar
                        * self.target_model(next_state.unsqueeze(0)).data.item()
                    )
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
        return sum(
            [
                pow(self.gamma, t * self.time_step * self.v_pref) * reward
                for t, reward in enumerate(rewards)
            ]
        )
