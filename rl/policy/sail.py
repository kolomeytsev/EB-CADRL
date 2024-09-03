import torch
import torch.nn as nn

from simulator.utils.action import ActionRot, ActionXY
from rl.policy.multi_human_rl import MultiHumanPolicy
from rl.utils.transform import MultiAgentTransform


class ExtendedNetwork(nn.Module):
    def __init__(self, num_adult, embedding_dim=64, hidden_dim=64, local_dim=32):
        super().__init__()
        self.num_adult = num_adult
        self.transform = MultiAgentTransform(num_adult)

        self.robot_encoder = nn.Sequential(
            nn.Linear(4, local_dim),
            nn.ReLU(inplace=True),
            nn.Linear(local_dim, local_dim),
            nn.ReLU(inplace=True),
        )

        self.adult_encoder = nn.Sequential(
            nn.Linear(4 * self.num_adult, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.adult_head = nn.Sequential(
            nn.Linear(hidden_dim, local_dim), nn.ReLU(inplace=True)
        )

        self.joint_embedding = nn.Sequential(
            nn.Linear(local_dim * 2, embedding_dim), nn.ReLU(inplace=True)
        )

        self.pairwise = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True)
        )

        self.planner = nn.Linear(hidden_dim, 2)

    def forward(self, robot_state, crowd_obsv):

        if len(robot_state.shape) < 2:
            robot_state = robot_state.unsqueeze(0)
            crowd_obsv = crowd_obsv.unsqueeze(0)

        # preprocessing
        emb_robot = self.robot_encoder(robot_state[:, :4])

        agent_state = self.transform.transform_frame(crowd_obsv)
        feat_adult = self.adult_encoder(agent_state)
        emb_adult = self.adult_head(feat_adult)

        emb_concat = torch.cat(
            [emb_robot.unsqueeze(1).repeat(1, self.num_adult, 1), emb_adult], axis=2
        )

        # embedding
        emb_pairwise = self.joint_embedding(emb_concat)

        # pairwise
        feat_pairwise = self.pairwise(emb_pairwise)

        # attention
        logit_pairwise = self.attention(emb_pairwise)
        score_pairwise = nn.functional.softmax(logit_pairwise, dim=1)

        # crowd
        feat_crowd = torch.sum(feat_pairwise * score_pairwise, dim=1)

        # planning
        reparam_robot_state = torch.cat(
            [robot_state[:, -2:] - robot_state[:, :2], robot_state[:, 2:4]], axis=1
        )
        feat_task = self.task_encoder(reparam_robot_state)

        feat_joint = self.joint_encoder(torch.cat([feat_task, feat_crowd], axis=1))
        action = self.planner(feat_joint)

        return action, feat_joint


class SAIL(MultiHumanPolicy):
    def __init__(self):
        super().__init__()
        self.name = "SAIL"

    def configure(self, config):
        self.set_common_parameters(config)
        self.multiagent_training = config.getboolean("sail", "multiagent_training")
        self.model = ExtendedNetwork(config.getint("sail", "adult_num"))

    def predict(self, state, env=None):
        if self.phase is None:
            raise AttributeError("Phase attribute has to be set!")
        if self.device is None:
            raise AttributeError("Device attributes has to be set!")
        if self.phase == "train" and self.epsilon is None:
            raise AttributeError("Epsilon attribute has to be set in training phase")

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == "holonomic" else ActionRot(0, 0)

        self.last_state = self.transform(state)

        action = self.model(self.last_state[0], self.last_state[1])[0].squeeze()

        if self.kinematics == "holonomic":
            return ActionXY(action[0].item(), action[1].item())
        else:
            return ActionRot(action[0].item(), action[1].item())

    def transform(self, state):
        """Transform state object to tensor input of RNN policy"""

        robot_state = torch.Tensor(
            [
                state.self_state.px,
                state.self_state.py,
                state.self_state.vx,
                state.self_state.vy,
                state.self_state.gx,
                state.self_state.gy,
            ]
        )

        num_adult = len(state.agent_states)
        agent_state = torch.empty([num_adult, 4])
        for k in range(num_adult):
            agent_state[k, 0] = state.agent_states[k].px
            agent_state[k, 1] = state.agent_states[k].py
            agent_state[k, 2] = state.agent_states[k].vx
            agent_state[k, 3] = state.agent_states[k].vy

        return [robot_state, agent_state]
