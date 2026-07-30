import torch.nn
from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, input_size, output_size, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.output_size = output_size
        if use_bn:
            self.decoder_head = torch.nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, self.output_size),
            )
        else:
            self.decoder_head = torch.nn.Sequential(
                nn.Linear(input_size, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, self.output_size),
            )


    def forward(self, x):
        """
        :param x: B x C
        :return: y: B x 4
        """
        return self.decoder_head(x)
