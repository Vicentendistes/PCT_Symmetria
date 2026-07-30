import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class TNet(nn.Module):
    """
    Transform Net: Network that learns a transformation matrix

    Args:
        in_dim (int): Input dimension
        k (int): Size of transformation matrix
    """

    def __init__(self, in_dim: int = 3, k: int = 3):
        super(TNet, self).__init__()
        self.in_dim = in_dim
        self.k = k

        self.shared_mlps = torch.nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
        )

        self.linear = torch.nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        :param x: Tensor of shape (batch_size, in_dim, num_points)
        :return: y: Tensor of shape (batch_size, out_dim)
        """
        batch_size = x.size()[0]
        x = torch.max(self.shared_mlps(x), dim=2).values
        x = self.linear(x)
        identity = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))
        ).view(1, self.k * self.k).repeat(batch_size, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, use_bn = False):
        super(PointNetEncoder, self).__init__()

        self.use_bn = use_bn

        if self.use_bn:
            self.input_transform = TNet(in_dim=3, k=3)
            self.feature_transform = TNet(in_dim=64, k=64)
            self.shared_mlps = torch.nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
            )
            self.shared_mlps_2 = torch.nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Conv1d(128, 256, 1),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 1024, 1),
            )

        else:
            self.input_transform = TNet(in_dim=3, k=3)
            self.feature_transform = TNet(in_dim=64, k=64)
            self.shared_mlps = torch.nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.LeakyReLU(),
            )
            self.shared_mlps_2 = torch.nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.LeakyReLU(),
                nn.Conv1d(128, 256, 1),
                nn.LeakyReLU(),
                nn.Conv1d(256, 1024, 1),
            )

    def forward(self, x):
        """
        :param x: Tensor of shape B x 3 x N of points.
        :return: Tensor of shape B x 1088 x N of features for each point.
                 The first 64 features are the local features, the next 1024 are global features.
        """
        input_trans = self.input_transform(x)
        input_trans = input_trans.reshape(-1, 3, 3)

        x0 = x.transpose(2, 1)
        x0 = torch.bmm(x0, input_trans)
        x0 = x0.transpose(2, 1)

        x1 = self.shared_mlps(x0)

        feat_trans = self.feature_transform(x1)
        feat_trans = feat_trans.reshape(-1, 64, 64)

        x1 = x1.transpose(2, 1)
        x1 = torch.bmm(x1, feat_trans)
        x1 = x1.transpose(2, 1)

        x2 = self.shared_mlps_2(x1)
        x2 = x2.transpose(2, 1)

        global_features = torch.max(x2, dim=1)[0]

        return global_features


if __name__ == "__main__":
    bs, sz = 16, 1024
    encoder = PointNetEncoder(use_bn=True)
    mock_x = torch.randn(bs, 3, sz)
    output = encoder.forward(mock_x)
    assert output.shape == (bs, 1024), f"Got {output.shape}"
    pytorch_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(pytorch_total_params)
