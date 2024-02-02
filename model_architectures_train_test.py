import torch


def train(device, model, optimizer, criterion, train_loader, val_loader, num_epochs=10):
    """
    Train and evaluate a pretrained model on dataset
    :param device: device to use
    :param model: model to train
    :param optimizer: optimizer to use
    :param criterion: criterion to evaluate
    :param train_loader: train dataset
    :param val_loader: validation dataset
    :param num_epochs: number of epochs
    :return:
    """
    for epoch in range(num_epochs):
        # train the model
        model.train()
        total_loss = 0.0

        # for each batch update gradients
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print loss for train
        avg_epoch_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}')

        # validate the model
        validation_loss, validation_accuracy = validate(model, val_loader, device, criterion)
        # print loss and accuracy for validation
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy * 100:.2f}%')

    return model


def validate(model, val_loader, device, criterion):
    """
    Validate a model on validation dataset
    :param model: model to validate
    :param val_loader: validation dataset
    :param device: device to use
    :param criterion: criterion to evaluate
    :return: Loss and accuracy of the validation dataset
    """
    # set model to evaluation
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # do not update gradients
    with torch.no_grad():
        # for each batch evaluate the loss and accuracy
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(val_loader), accuracy


def test(model, test_loader, device, criterion):
    """
    Test a model on test dataset
    :param model: model to test
    :param test_loader: test dataset
    :param device: devide to use
    :param criterion: criterion to evaluate
    :return: None
    """
    # set model to evaluation
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # do not update gradients
    with torch.no_grad():
        # for each batch evaluate the loss and accuracy
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    # print the results
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Test Accuracy: {accuracy * 100:.2f}%")
    print(f"# unique correct samples: {correct}")
    print(f"# unique errors: {total - correct}")
