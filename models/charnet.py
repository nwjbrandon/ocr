import torch.nn as nn
import torchvision

import models.resnet as m_resnet


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CharNet(nn.Module):
    def __init__(self, cfg, bilinear=False):
        super(CharNet, self).__init__()
        self.in_channels = cfg["model"]["in_channels"]

        self.inc = DownConv(self.in_channels, 3)
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.flat = nn.Flatten(start_dim=2, end_dim=-1)

        self.fc = nn.Linear(40, cfg["model"]["n_char"])
        self.outc = nn.Conv1d(512, cfg["model"]["n_classes"], kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.backbone(x1)
        x3 = self.flat(x2)
        x4 = self.fc(x3)
        logits = self.outc(x4)
        return logits


class CharSeqNet(nn.Module):
    def __init__(self, cfg):
        super(CharSeqNet, self).__init__()
        self.in_channels = cfg["model"]["in_channels"]
        self.T = cfg["model"]["n_char"] * 2 + 1

        self.backbone = m_resnet.create_resnet18()
        self.flat = nn.Flatten(start_dim=1, end_dim=2)

        self.t_out = nn.Sequential(nn.Linear(13, self.T),)

        self.c_out = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, cfg["model"]["n_classes"], kernel_size=1),
        )
        self.sm = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flat(x)
        x = self.t_out(x)
        x = self.c_out(x)
        x = x.permute(0, 2, 1)
        logits = self.sm(x)
        # logits.shape: B, T, C (64, 21, 24)
        return logits


class CharSeqGruNet(nn.Module):
    def __init__(self, cfg):
        super(CharSeqGruNet, self).__init__()
        self.in_channels = cfg["model"]["in_channels"]
        self.gru_hidden_size = 128
        self.gru_num_layers = 2
        self.gru_in_channels = 1024
        self.T = cfg["model"]["n_char"] * 2 - 1

        self.backbone = m_resnet.create_resnet18()
        self.t_out = nn.Conv2d(13, self.T, kernel_size=1)
        self.flat = nn.Flatten(start_dim=2, end_dim=3)
        self.gru = nn.GRU(
            self.gru_in_channels,
            self.gru_hidden_size,
            self.gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.c_out = nn.Linear(
            self.gru_hidden_size * self.gru_num_layers,
            cfg["model"]["n_classes"],
        )
        self.sm = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 2, 1)
        x = self.t_out(x)
        x = self.flat(x)
        x, _ = self.gru(x)
        x = self.c_out(x)
        logits = self.sm(x)
        # logits.shape: B, T, C (64, 21, 24)
        return logits
