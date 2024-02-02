import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from FishDataset import FishDataset
from model_architectures_train_test import train, test
from model_architectures_utils import get_all_models
from utils import get_train_test_split


def train_models(models, dataset_path, device, num_epochs, batch_size=32, learning_rate=0.001,
                 test_ratio=0.2):
    """
    Train models on dataset
    :param models: models to train
    :param dataset_path: path to images
    :param device: device to use
    :param num_epochs: number of epochs to train each model
    :param batch_size: batch size for retrieving the images
    :param learning_rate: learning rate to use in models training
    :param test_ratio: test and validation ratio
    :return: trained models and datasets
    """
    # get pd dataframes for each part of the dataset
    fish_train, fish_val, fish_test = get_train_test_split(test_ratio=test_ratio, dataset_path=dataset_path,
                                                           split_seed=12345, get_val_data=True)
    # transform the images (resizing)
    transform = transforms.Compose([
        transforms.Resize((590, 445)),
        transforms.ToTensor(), ])

    # get datasets
    train_dataset = FishDataset(annotations=fish_train, img_dir=dataset_path, transform=transform, get_path=False)
    val_dataset = FishDataset(annotations=fish_val, img_dir=dataset_path, transform=transform, get_path=False)
    test_dataset = FishDataset(annotations=fish_test, img_dir=dataset_path, transform=transform, get_path=False)

    # get loaders from datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    trained_models = []
    for model, model_name in models:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.to(device)

        print('Model: ', model_name)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")

        # train and validate
        start_time = time.time()
        trained_model = train(device, model, optimizer, criterion, train_loader, val_loader, num_epochs=num_epochs)
        train_end_time = time.time()
        print(f"Training time: {train_end_time - start_time:.2f} seconds")

        # test the model
        test(trained_model, test_loader, device, criterion)
        test_end_time = time.time()
        print(f"Testing time: {test_end_time - train_end_time:.2f} seconds")

        trained_models.append((trained_model, model_name))
    return trained_models, (train_loader, val_loader, test_loader)


if __name__ == '__main__':
    fish_dataset_path = 'NA_Fish_Dataset'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = get_all_models(9)
    train_models(models, fish_dataset_path, device, num_epochs=10, batch_size=32, learning_rate=0.001)
