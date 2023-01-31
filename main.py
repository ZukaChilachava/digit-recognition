import sys
import time
import os.path as pt

import torch
import torch.optim as optim
from Trainer import ModelTrainer
from Dataset import LetterDataset
from LetterPredictionModel import MultiClassModel


def test(dataset: LetterDataset, model: MultiClassModel):
    sample_number = len(dataset)
    outputs, features = dataset.y, dataset.x

    with torch.no_grad():
        prediction_magnitudes = model(features)

        predictions = torch.argmax(prediction_magnitudes, 1)
        # boolean array -> summing True value -> converting tensor(val) to val
        correctly_predicted = (predictions == outputs).sum().item()

    print("Accuracy:", 100 * correctly_predicted / sample_number, "%")


def run_model(optimizer, model, optim_name):
    trainer = ModelTrainer(optimizer,
                           train_dataset,
                           training_samples,
                           model)

    start = time.perf_counter()
    trainer.train()
    end = time.perf_counter()
    print(f"{optim_name} optimizer speed: {end - start:0.4f}s")
    test(test_dataset, model)


if __name__ == '__main__':
    total_samples = 20000
    training_samples = 16000
    path_to_root = sys.path[1]
    path_to_dataset = pt.join(path_to_root, "Dataset", "letter-recognition.data")

    train_dataset = LetterDataset(training_samples, 0, path=path_to_dataset)
    test_dataset = LetterDataset(total_samples - training_samples,
                                 training_samples,
                                 path=path_to_dataset)

    num_letters = 27
    learning_rate = 0.01

    adam_model = MultiClassModel(input_size=train_dataset[0][1].shape[0],
                                 num_classes=num_letters)
    run_model(optim.Adam(adam_model.parameters(), lr=learning_rate), adam_model, "Adam")

    sgd_model = MultiClassModel(input_size=train_dataset[0][1].shape[0],
                                 num_classes=num_letters)
    run_model(optim.SGD(sgd_model.parameters(), lr=0.1), sgd_model, "SGD")
