from pathlib import Path
from datetime import datetime

import yaml

import torch
from torch import nn

from torchnet.logger import VisdomPlotLogger


class CheckpointManager(object):
    """A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Parameters
    ----------
    config : yaml
        The configuration file
    checkpoint_dirpath : list
        Path to an empty or non-existent directory to save checkpoints.
    visual_log_state : bool, optional
        Whether to log using visdom or not, by default False
    model_log_state : bool, optional
        Whether to log the pytroch model or not, by default False
    mode : str, optional
        Mode used to find the best metric and best model to store,
        by default "max"
    step_size : int, optional
        Stepsize at which the models are stored, by default 5

    Example
    --------
    >>> checkpoint_dirpath = ['trained_models', 'iteratio_1']
    >>> ckpt_manager = CheckpointManager(config, checkpoint_dirpath,
                       model_log_state=True)
    >>> for epoch in range(20):
    ...     train(...)
    ...     val_accuracy = validate(...)
    ...     ckpt_manager.step(val_accuracy)
    """
    def __init__(self,
                 config,
                 checkpoint_dirpath,
                 visual_log_state=False,
                 model_log_state=False,
                 mode="max",
                 step_size=5):

        # Parameters
        self.visual_log_state = visual_log_state
        self.model_log_state = model_log_state
        self.mode = mode
        self.config = config
        self.step_size = step_size

        # Computing device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        if isinstance(checkpoint_dirpath, list):
            checkpoint_dirpath = '/'.join(checkpoint_dirpath)
        self.ckpt_dirpath = Path(config['logs_path'] + checkpoint_dirpath)
        self._init_directory()
        self._setup_visual_logger()

    def _init_directory(self):
        """Initialize empty checkpoint directory and record commit SHA
        in it. Extend this method to do more fancy things at start of
        experiment, for example saving hyper-parameters as a YAML file.
        """

        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        # Save the config file
        with open(self.ckpt_dirpath / 'config.yaml', 'w') as f:
            yaml.dump(self.config,
                      f,
                      default_flow_style=False,
                      sort_keys=False)
        f.close()

        # Save time also
        time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M")
        with open(self.ckpt_dirpath / 'time.txt', "a") as f:
            f.write(time_stamp + '\n')
        f.close()
        return None

    def _setup_visual_logger(self):
        """Return a pytorch tnt visual loggger.

        Parameters
        ----------
        title : str
            A title to describe the logging.

        Returns
        -------
        type
            pytorch visual logger.

        """
        self.visual_logger = VisdomPlotLogger('line',
                                              opts=dict(
                                                  legend=None,
                                                  xlabel='Epochs',
                                                  ylabel='Criterion',
                                              ),
                                              env=None)
        return None

    def _model_state_dict(self, model):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(model, nn.DataParallel):
            return model.module.state_dict()
        else:
            return model.state_dict()

    def _save_current(self, model, optimizer, metric, current_epoch, loss):
        # Save the current model
        torch.save(
            {
                "model_weights": self._model_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "metric": metric,
                "loss": loss,
                "epoch": current_epoch
            }, self.ckpt_dirpath / f"checkpoint_{current_epoch}.pth")
        return None

    def _save_best(self, model, optimizer, metric, current_epoch, loss):
        """Save best performing checkpoint observed so far."""
        # Default value
        is_best = False

        if self.mode == 'min':
            if metric['testing'] <= self.best_metric:
                is_best = True
        else:
            if metric['testing'] >= self.best_metric:
                is_best = True

        # Update the best and save the model
        if is_best:
            self.best_metric = metric['testing']
            torch.save(
                {
                    "model_weights": self._model_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "metric": metric,
                    "loss": loss,
                    "epoch": current_epoch
                }, self.ckpt_dirpath / "checkpoint_best.pth")
        return None

    def load_saved_checkpoint(self, epoch_id, best=False):
        if best:
            read_path = self.ckpt_dirpath / "checkpoint_best'.pth"
        else:
            read_path = self.ckpt_dirpath / f"checkpoint_{epoch_id}.pth"

        # Read the saved model
        saved_model = torch.load(read_path, map_location=self.device)
        return saved_model

    def initial_save(self, model, optimizer, criterion):
        torch.save(
            {
                "model": model,
                "optimizer": optimizer,
                "criterion": criterion,
            }, self.ckpt_dirpath / "initial_state.pth")

    def visual_log(self, current_epoch, metric, title='', env='main'):
        if self.visual_log_state and not current_epoch % self.step_size:
            # Change the legend
            self.visual_logger.opts['legend'] = list(metric.keys())
            self.visual_logger.opts['title'] = title
            self.visual_logger.env = env
            self.visual_logger.log(current_epoch, list(metric.values()))
        return None

    def model_log(self, model, optimizer, metric, current_epoch, loss=None):
        """Save checkpoint if step size conditions meet, and update
        best checkpoint based on metric and mode.
        """
        if self.model_log_state:
            if current_epoch == 0:
                self.best_metric = metric['testing']

            # Check if the current model is the best
            self._save_best(model, optimizer, metric, current_epoch, loss)

            if not current_epoch % self.step_size:
                self._save_current(model, optimizer, metric, current_epoch,
                                   loss)


def load_saved_checkpoint(config,
                          epoch_id,
                          checkpoint_dirpath,
                          best=False,
                          apply_weights=False):
    # Directory path
    ckpt_dirpath = Path(config['logs_path'] + '/'.join(checkpoint_dirpath))

    # Computing device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if best:
        read_path = ckpt_dirpath / "checkpoint_best.pth"
    else:
        read_path = ckpt_dirpath / f"checkpoint_{epoch_id}.pth"

    # Read the saved model
    saved_model = torch.load(read_path, map_location=device)

    if apply_weights:
        read_path = ckpt_dirpath / "initial_state.pth"
        model = torch.load(read_path, map_location=device)['model']
        model.load_state_dict(saved_model['model_weights'])
        return model
    else:
        return saved_model


def calculate_accuracy(model, data_iterator, key=None):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : dict
        A dictionary containing data iterator object.
    key : str
        A key to select which dataset to evaluate

    Returns
    -------
    float
        accuracy of classification for the given key.

    """
    if key is None:
        keys = data_iterator.keys()
    else:
        keys = [key]  # make it as a list

    accuracy = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for key in keys:
            total = 0
            length = 0
            for x, y in data_iterator[key]:
                model.eval()
                out_put, _ = model(x.to(device))
                out_put = out_put.cpu().detach()
                total += (out_put.argmax(dim=1) == y.argmax(
                    dim=1)).float().sum()
                length += len(y)
            accuracy[key] = (total / length).numpy()
    return accuracy


def calculate_rms(model, data_iterator, criterion, key=None):
    if key is None:
        keys = data_iterator.keys()
    else:
        keys = [key]  # make it as a list

    avg_loss = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for key in keys:
            loss = 0
            length = 0
            for x, y in data_iterator[key]:
                model.eval()
                predicted, _ = model(x.to(device))
                loss += torch.sqrt(criterion(predicted, y.to(device)))
                length += 1
            avg_loss[key] = (loss.cpu().detach() / length).numpy()
    return avg_loss


def calcuate_loss(model, data_iterator, criterion, key=None):
    if key is None:
        keys = data_iterator.keys()
    else:
        keys = [key]  # make it as a list

    avg_loss = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for key in keys:
            loss = 0
            length = 0
            for x, y in data_iterator[key]:
                model.eval()
                out_put, _ = model(x.to(device))
                y = (torch.max(y, dim=1)[1]).to(device)
                loss += criterion(out_put, y)
                length += 1
            avg_loss[key] = (loss.cpu().detach() / length).numpy()
    return avg_loss


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param = parameter.numel()
            total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params
