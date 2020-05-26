import torch
import torch.nn as nn


class ShallowERPNet(nn.Module):
    """Convolution neural network class for EMG classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT, config):
        super(ShallowERPNet, self).__init__()
        # Network blocks
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(1, 15), stride=1, bias=True),
            nn.Conv2d(2, 2, kernel_size=(32, 8), stride=1, bias=True),
            nn.BatchNorm2d(2, momentum=0.1, affine=True))
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(2, OUTPUT, kernel_size=(1, 8), stride=1, bias=True))

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        out = self.net_1(x)

        out = out * out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)

        out = self.net_2(out)
        out = torch.log_softmax(out, dim=1)
        out = torch.squeeze(out)

        return out


class ShiftScaleERPNet(nn.Module):
    """Convolution neural network class for EMG classification.

    Parameters
    ----------
    OUTPUT : int
        Number of classes.

    Attributes
    ----------
    net_1 : pytorch Sequential
        Convolution neural network class for eeg classification.
    pool : pytorch pooling
        Pooling layer.
    net_2 : pytorch Sequential
        Classification convolution layer.

    """
    def __init__(self, OUTPUT, config):
        super(ShiftScaleERPNet, self).__init__()
        # Configuration of EMGsignals
        self.epoch_length = config['epoch_length']
        self.s_freq = config['sfreq']
        self.n_electrodes = config['n_electrodes']

        # Network blocks
        self.mean_net = nn.Linear(self.n_electrodes,
                                  self.n_electrodes,
                                  bias=False)
        self.std_net = nn.Linear(self.n_electrodes,
                                 self.n_electrodes,
                                 bias=False)
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(1, 15), stride=1),
            nn.Conv2d(5, 5, kernel_size=(32, 8), stride=1))
        # nn.BatchNorm2d(10, momentum=0.1, affine=True))
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(5, OUTPUT, kernel_size=(1, 8), stride=1))

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        # The normalisation network
        shift = self.mean_net(x.mean(dim=3))
        x_shifted = x - shift[:, :, :, None]
        scale = self.std_net(x_shifted.std(dim=3))
        x_scaled = x_shifted * scale[:, :, :, None]

        # The convolution network
        out = self.net_1(x_scaled)

        out = out * out
        out = self.pool(out)

        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)

        out = self.net_2(out)
        out = torch.log_softmax(out, dim=1)
        out = torch.squeeze(out)

        return out
