import torch
import torch.nn as nn


class ShallowNet(nn.Module):
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
        super(ShallowNet, self).__init__()

        # Network blocks
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), stride=1),
            nn.Conv2d(32, 32, kernel_size=(20, 5), stride=1))
        self.pool = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 5))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(32, OUTPUT, kernel_size=(1, 10), stride=1),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        out = self.net_1(x)

        out = out * out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        # out = self.dropout(out)

        out = self.net_2(out)
        out = torch.squeeze(out)

        return out


class ShiftScaleEEGClassify(nn.Module):
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
        super(ShiftScaleEEGClassify, self).__init__()
        self.n_electrodes = config['n_electrodes']

        # Network blocks
        self.mean_net = nn.Linear(self.n_electrodes,
                                  self.n_electrodes,
                                  bias=False)
        self.std_net = nn.Linear(self.n_electrodes,
                                 self.n_electrodes,
                                 bias=False)
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 30), stride=1),
            nn.Conv2d(10, 10, kernel_size=(20, 40), stride=1))
        self.batch_norm = nn.BatchNorm2d(20,
                                         momentum=0.1,
                                         affine=True,
                                         eps=1e-05)
        self.pool = nn.AvgPool2d(kernel_size=(1, 150), stride=(1, 30))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(10, OUTPUT, kernel_size=(1, 10), stride=1))
        self.full = nn.Linear(in_features=40, out_features=OUTPUT)

    def forward(self, x):
        x = x[:, None, :, :]  # Add the extra dimension
        # The normalisation network
        shift = self.mean_net(x.mean(dim=3))
        x_shifted = x - shift[:, :, :, None]
        scale = self.std_net(x_shifted.std(dim=3))
        x_scaled = x_shifted * scale[:, :, :, None]

        # The convolution network
        out = self.net_1(x_scaled)

        # Squaring and log
        out = out * out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)

        # Softmax
        out = self.net_2(out)
        prob = torch.softmax(out, dim=1)
        log_prob = torch.log_softmax(out, dim=1)

        # Flatten them matching the number of examples in the batch
        prob = torch.squeeze(prob)
        log_prob = torch.squeeze(log_prob)

        return log_prob, prob


class ShiftScaleEEGRegress(nn.Module):
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
        super(ShiftScaleEEGRegress, self).__init__()
        self.n_electrodes = config['n_electrodes']

        # Network blocks
        self.mean_net = nn.Sequential(
            nn.Linear(self.n_electrodes, self.n_electrodes))
        self.std_net = nn.Sequential(
            nn.Linear(self.n_electrodes, self.n_electrodes))
        self.net_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 15), stride=1),
            nn.Conv2d(10, 10, kernel_size=(20, 20), stride=1),
            nn.BatchNorm2d(10, momentum=0.1, affine=True, eps=1e-05))
        self.batch_norm = nn.BatchNorm2d(20,
                                         momentum=0.1,
                                         affine=True,
                                         eps=1e-05)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=config['DROP_OUT'])
        self.net_2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(1, 10), stride=1))
        self.full = nn.Sequential(
            nn.Linear(in_features=10, out_features=5), nn.Tanh(),
            nn.Linear(in_features=5, out_features=OUTPUT))
        self.loss = nn.MSELoss()

    def forward(self, x):
        # The normalisation network
        x = x[:, None, :, :]  # Add the extra dimension
        shift = self.mean_net(x.mean(dim=3))
        x_shifted = x - shift[:, :, :, None]
        scale = self.std_net(x_shifted.std(dim=3))
        x_scaled = x_shifted * scale[:, :, :, None]

        # The convolution network
        out = self.net_1(x_scaled)
        # out = self.net_1(x)

        # Squaring and log
        out = out * out
        out = self.pool(out)
        out = torch.log(torch.clamp(out, min=1e-6))
        out = self.dropout(out)

        # Final convolution
        out = self.net_2(out)

        # predicted = predicted.view(predicted.size(0), -1)
        predicted = self.full(out.view(out.size(0), -1))

        # Squared error
        squared_error = None

        return predicted, squared_error
