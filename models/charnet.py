import torch.nn as nn


class BidirectionalGRU(nn.Module):
    def __init__(self, nIn, nHidden, nOut, n_layers):
        super().__init__()
        self.rnn = nn.LSTM(
            nIn,
            nHidden,
            bidirectional=True,
            batch_first=True,
            num_layers=n_layers,
        )
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, inp):
        recurrent, _ = self.rnn(inp)
        (B, T, H,) = recurrent.size()
        t_rec = recurrent.reshape(T * B, H)
        output = self.embedding(t_rec)
        output = output.reshape(T, B, -1)
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        return x


class CharNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg["model"]["in_channels"]
        self.n_char = cfg["model"]["n_char"]
        self.rnn_hidden_size = 128
        self.rnn_num_layers = 2
        self.rnn_in_channels = 768
        self.T = 25

        self.conv1 = Conv(self.in_channels, 64)
        self.conv2 = Conv(64, 128)
        self.conv3 = Conv(128, 256)
        self.t_out = nn.Linear(14, self.n_char * 2 + 1)

        self.flat = nn.Flatten(start_dim=2, end_dim=3)
        self.rnn = nn.Sequential(
            BidirectionalGRU(
                self.rnn_in_channels,
                self.rnn_hidden_size,
                self.rnn_hidden_size,
                self.rnn_num_layers,
            ),
            BidirectionalGRU(
                self.rnn_hidden_size,
                self.rnn_hidden_size,
                cfg["model"]["n_classes"],
                self.rnn_num_layers,
            ),
        )
        self.sm = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.t_out(x)
        x = x.permute(0, 3, 2, 1)
        x = self.flat(x)
        x = self.rnn(x)
        logits = self.sm(x)
        # B, T, C (64, 21, 11)
        return logits
