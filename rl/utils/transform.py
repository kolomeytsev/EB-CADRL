import torch


class MultiAgentTransform():

    def __init__(self, num_adult, state_dim=4):
        self.num_adult = num_adult
        self.mask = torch.ones(num_adult, num_adult, state_dim).bool()
        for k in range(num_adult): self.mask[k, k] = False      

    def transform_frame(self, frame):
        bs = frame.shape[0]
        # [length, num_adult, num_adult, state_dim]
        compare = frame.unsqueeze(1) - frame.unsqueeze(2)
        # [length, num_adult, (num_adult-1) * state_dim]
        relative = torch.masked_select(compare, self.mask.repeat(bs, 1, 1, 1)).reshape(bs, self.num_adult, -1)
        state = torch.cat([frame, relative], axis=2)
        return state
