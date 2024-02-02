from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import nn
from torchvision.models import resnet18, resnet50, densenet121, vgg19


def get_all_models(num_classes):
    """
    Get a list of all models
    :param num_classes: number of classes in dataset
    :return: models list
    """
    # vgg19
    vgg19_model = vgg19(pretrained=True)
    vgg19_model.classifier[6] = nn.Linear(vgg19_model.classifier[6].in_features, num_classes)

    # resnet18
    resnet18_model = resnet18(pretrained=True)
    resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, num_classes)

    # densenet121
    densenet121_model = densenet121(pretrained=True)
    densenet121_model.classifier = nn.Linear(densenet121_model.classifier.in_features, num_classes)

    # resnet50
    resnet50_model = resnet50(pretrained=True)
    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)

    return [(vgg19_model, 'VGG19'), (resnet18_model, 'ResNet18'), (densenet121_model, 'DenseNet121'),
            (resnet50(pretrained=True), 'ResNet50')]


def get_resnet18(num_classes):
    """
    Get resnet18 model
    :param num_classes: number of classes in dataset
    :return: resnet18 model
    """
    # resnet18
    resnet18_model = resnet18(pretrained=True)
    resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, num_classes)

    return [(resnet18_model, 'ResNet18')]


def get_classifiers():
    """
    Get a list of classifiers
    :return: classifiers list
    """
    rf = (RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest')
    svc = (SVC(kernel='linear', C=1.0, random_state=42), 'SVM')
    knn = (KNeighborsClassifier(n_neighbors=5), 'KNN')
    return [rf, svc, knn]
