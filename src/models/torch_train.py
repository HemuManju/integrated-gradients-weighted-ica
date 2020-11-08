import torch

from .utils import calculate_accuracy, calculate_rms


class TorchTrainer(object):
    def __init__(self,
                 config,
                 data_iterator,
                 model,
                 criterion,
                 optimizer,
                 ckpt_manager,
                 task_type='classification'):
        super(TorchTrainer, self).__init__()
        # Computing device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialise the parameters
        self.data_iterator = data_iterator
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.ckpt_manager = ckpt_manager
        self.config = config
        self.task_type = task_type

        # Save initial state
        self.ckpt_manager.initial_save(self.model, self.optimizer,
                                       self.criterion)

    def train(self):
        for epoch in range(self.config['NUM_EPOCHS']):
            for x_batch, y_batch in self.data_iterator['training']:
                # Send the input and targets to gpu
                x_batch = x_batch.to(self.device)
                if self.task_type == 'classification':
                    y_batch = (torch.max(y_batch, dim=1)[1]).to(self.device)
                else:
                    y_batch = y_batch.to(self.device)

                # Forward pass
                log_prob, prob = self.model(x_batch)
                loss = self.criterion(log_prob, y_batch)

                # Backward pass and optimize
                self.optimizer.zero_grad()  # For batch gradient optimisation
                loss.backward()
                self.optimizer.step()

            # Calculate accuracy and log them
            if self.task_type == 'classification':
                metrics = calculate_accuracy(self.model, self.data_iterator)
            else:

                metrics = calculate_rms(self.model, self.data_iterator,
                                        self.criterion)

            self.ckpt_manager.model_log(self.model, self.optimizer, metrics,
                                        epoch)

            # Visual logging
            if self.ckpt_manager.visual_logger:
                self.ckpt_manager.visual_log(epoch, metrics)

        return None

    def transfer(self):
        # Switch off the gradient by default to all parameters
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Train only the last layers used
        for parameter in self.model.full.parameters():
            parameter.requires_grad = True

        for epoch in range(self.config['NUM_EPOCHS']):
            for x_batch, y_batch in self.data_iterator['training']:
                # Send the input and targets to gpu
                x_batch = x_batch.to(self.device)
                if self.task_type == 'classification':
                    y_batch = (torch.max(y_batch, dim=1)[1]).to(self.device)
                else:
                    y_batch = y_batch.to(self.device)

                # Forward pass
                log_prob, prob = self.model(x_batch)
                loss = self.criterion(log_prob, y_batch)

                # Backward pass and optimize
                self.optimizer.zero_grad()  # For batch gradient optimisation
                loss.backward()
                self.optimizer.step()

            # Calculate accuracy and log them
            if self.task_type == 'classification':
                metrics = calculate_accuracy(self.model, self.data_iterator)
            else:
                metrics = calculate_rms(self.model, self.data_iterator,
                                        self.criterion)

            self.ckpt_manager.model_log(self.model, self.optimizer, metrics,
                                        epoch)

            # Visual logging
            if self.ckpt_manager.visual_logger:
                self.ckpt_manager.visual_log(epoch, metrics)

        return None
