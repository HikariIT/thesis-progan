import torch
import torch.nn as nn


class MiniBatchStdDev(nn.Module):

    group_size: int

    def __init__(self, mini_batch_size: int = 4):
        super().__init__()
        self.group_size = mini_batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        sub_group_size = min(size[0], self.group_size)

        if size[0] % sub_group_size != 0:
            sub_group_size = size[0]

        no_groups = size[0] // sub_group_size
        _, filters, height, width = size
        eps = 1e-8

        if sub_group_size > 1:
            y = x.view(-1, sub_group_size, filters, height, width)
            y = torch.sqrt(torch.var(y, 1)+ eps)
            y = y.view(no_groups, -1)
            y = torch.mean(y, 1).view(no_groups, 1)
            y = y.expand(no_groups, width * height).view(no_groups, 1, 1, width, height)
            y = y.expand(no_groups, sub_group_size, -1, -1, -1)
            y = y.contiguous().view(-1, 1, height, width)
        else:
            y = torch.zeros(size[0], 1, height, width, device=x.device)

        return torch.cat([x, y], 1)