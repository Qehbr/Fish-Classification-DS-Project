import numpy as np
import torch
from sklearn.metrics import accuracy_score
from model_architectures_main import train_models
from model_architectures_utils import get_resnet18, get_classifiers


def evaluate_classifiers(models, classifiers, train_loader, val_loader, test_loader):
    """
    Evaluate the classifiers on pretrained models
    :param models: pretrained models
    :param classifiers: classifiers to evaluate
    :param train_loader: train dataset
    :param val_loader: validation dataset
    :param test_loader: test dataset
    :return:
    """
    for model, model_name in models:
        # remove last classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])

        # extract features
        train_features, train_labels = extract_features(train_loader, model)
        val_features, val_labels = extract_features(val_loader, model)
        test_features, test_labels = extract_features(test_loader, model)

        for classifier, classifier_name in classifiers:
            # use features to train classifier
            classifier.fit(train_features, train_labels)

            # get predictions and print the results
            val_predictions = classifier.predict(val_features)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            print(f'{model_name} + {classifier_name} Validation Accuracy: {val_accuracy * 100:.2f}%')

            test_predictions = classifier.predict(test_features)
            test_accuracy = accuracy_score(test_labels, test_predictions)
            print(f'{model_name} + {classifier_name} Test Accuracy: {test_accuracy * 100:.2f}%')


def extract_features(loader, model):
    """
    Extract features from a dataset using feature extractor model
    :param loader: dataset
    :param model: feature extractor model
    :return:
    """
    features, labels = [], []
    # set model to evaluation
    model.eval()
    # no gradients updating
    with torch.no_grad():
        for inputs, labels_batch in loader:
            inputs = inputs.to(device)

            # forward pass the images and get features
            features_batch = model(inputs)
            features_batch = features_batch.view(features_batch.size(0), -1)
            features.append(features_batch.cpu().numpy())
            labels.append(labels_batch.cpu().numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    return features, labels


if __name__ == '__main__':
    fish_dataset_path = 'NA_Fish_Dataset'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = get_resnet18(9)
    trained_models, (train_loader, val_loader, test_loader) = train_models(models, fish_dataset_path, device,
                                                                           num_epochs=10, batch_size=32,
                                                                           learning_rate=0.001)

    classifiers = get_classifiers()
    evaluate_classifiers(trained_models, classifiers, train_loader, val_loader, test_loader)
