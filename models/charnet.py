""" Full assembly of the parts to form the complete network """


import torch.nn as nn
import torchvision


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

        self.fc = nn.Linear(91, cfg["model"]["n_char"])
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
        self.gru_hidden_size = 128
        self.gru_num_layers = 2

        self.inc = DownConv(self.in_channels, 3)
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.fc1 = nn.Linear(13, cfg["model"]["n_char"] * 2 + 1,)

        self.flat = nn.Flatten(start_dim=2, end_dim=-1)
        self.gru = nn.GRU(
            2048,
            self.gru_hidden_size,
            self.gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc2 = nn.Linear(
            self.gru_hidden_size * self.gru_num_layers,
            cfg["model"]["n_classes"],
        )
        self.sm = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.backbone(x1)
        x3 = self.fc1(x2)
        x3 = x3.permute(0, 3, 2, 1)
        x3 = self.flat(x3)
        x4, _ = self.gru(x3)
        x5 = self.fc2(x4)
        logits = self.sm(x5)
        return logits
