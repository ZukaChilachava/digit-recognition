import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from LetterPredictionModel import MultiClassModel


@dataclass
class ModelTrainer:
    optimizer: optim
    dataset: Dataset
    training_samples: int
    model: MultiClassModel
    num_epochs: int = 100
    batch_size: int = 4096
    loss_fun: nn.modules.loss = nn.CrossEntropyLoss()

    def train(self):
        _, features = self.dataset[:self.training_samples]
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            net_loss = 0
            for i, (labels, inputs) in enumerate(dataloader):
                # predictions of a current model
                outputs = self.model(inputs)
                # loss for current inputs according to correct labels
                loss = self.loss_fun(outputs, labels.long())

                net_loss += loss.item()
                # erase previous gradient
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        return self.model
