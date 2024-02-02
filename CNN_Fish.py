import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
import lightning as L
import torch.nn.functional as F
from FishDataset import FishDataset
from PIL import Image


class CNN_Fish(L.LightningModule):
    """
    CNN model for Fish classification
    """

    def __init__(self, annotations_train, annotations_val, annotations_test, dataset_path, fold_index, num_classes,
                 learning_rate=None, batch_size=16, augmentation=False, normalization=False, ITA=False):
        """
        Class constructor
        :param annotations_train: pandas dataframe containing training annotations (image path / label encoded class)
        :param annotations_val: pandas dataframe containing validation annotations (image path / label encoded class)
        :param annotations_test: pandas dataframe containing test annotations (image path / label encoded class)
        :param dataset_path: dataset path with images
        :param fold_index: current fold index for k-fold cross validation
        :param num_classes: number of classes in dataset
        :param learning_rate: learning rate (if None scheduler will be used)
        :param batch_size: batch size (if ITA used batch size will be multiple by 8 - number of augmented images)
        :param augmentation: use augmentation or not
        :param normalization: use normalization or not
        :param ITA: use Inference Time Augmentation or not
        """
        super().__init__()

        # variables about dataset
        self.dataset_path = dataset_path
        self.annotations_train = annotations_train
        self.annotations_val = annotations_val
        self.annotations_test = annotations_test

        # config
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.ITA = ITA
        self.fold_index = fold_index

        # transforms for images (resizing and normalization if true)
        if normalization:
            self.image_transform = transforms.Compose([
                transforms.Resize((590, 445)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((590, 445)),
                transforms.ToTensor(),
            ])

        # model layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 32, 3, padding='same')
        self.conv4 = nn.Conv2d(32, 64, 3, padding='same')
        self.linear1 = nn.Linear(147 * 111 * 64, 50)
        self.linear2 = nn.Linear(50, self.num_classes)
        self.mp = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # accuracy metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # validation metrics
        self.train_losses_batches = []
        self.val_losses_batches = []

        # history of accuracy and validation for each epoch
        self.train_accs = []
        self.train_losses = []
        self.val_accs = []
        self.val_losses = []

        # correct predicted images / incorrect predicted images / uncertain predictions
        self.correct_predictions = []
        self.incorrect_predictions = []
        self.uncertain_predictions = []

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input
        :return: output of the model
        """

        # convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.mp(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.mp(x)

        # fully connected layers
        x = x.view(-1, 147 * 111 * 64)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        """
        Train the model
        :param batch: batch of images to train from
        :param batch_idx: current batch index
        :return: loss of the current batch
        """
        x, y = batch[0], batch[1].to(torch.int64)

        # if augmentation used
        if self.augmentation:

            # augment images
            augmented_images = []
            augmented_labels = []
            # create 7 new images for each image in batch with rotation 45 degrees
            range_augmentation = range(0, 360, 45)
            for i in range(len(x)):
                image = x[i]
                label = y[i]
                for angle in range_augmentation:
                    augmented_images.append(transforms.RandomRotation(angle)(image))
                    augmented_labels.append(label)

            # forward pass each image
            augmented_logits = []
            for image in augmented_images:
                augmented_logits.append(self.forward(image))
            logits = torch.cat(augmented_logits, dim=0)
            y = torch.stack(augmented_labels, dim=0)
        else:
            # forward pass images
            logits = self.forward(x)

        # calculate loss
        loss = F.nll_loss(logits, y)
        self.train_losses_batches.append(loss)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)

        # log the results
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """
        Called at the end of epoch, used for saving accuracy and loss history
        :return: None
        """
        # save accuracy for this epoch
        self.train_accs.append(self.train_accuracy.compute())
        # calculate and save loss for this epoch
        self.train_losses.append(sum(self.train_losses_batches) / len(self.train_losses_batches))
        self.train_losses_batches.clear()

    def validation_step(self, batch, batch_idx):
        """
        Validate the model
        :param batch: batch of images to validate
        :param batch_idx: current batch index
        :return: None
        """
        x, y = batch[0], batch[1].to(torch.int64)

        # forward pass images
        logits = self.forward(x)

        # calculate loss
        loss = F.nll_loss(logits, y)
        self.val_losses_batches.append(loss)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # log the results
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, )

    def on_validation_epoch_end(self):
        """
        Called at the end of epoch, used for saving accuracy and loss history
        :return: None
        """
        # save accuracy for this epoch
        self.val_accs.append(self.val_accuracy.compute())
        # calculate and save loss for this epoch
        self.val_losses.append(sum(self.val_losses_batches) / len(self.val_losses_batches))
        self.val_losses_batches.clear()

    def test_step(self, batch, batch_idx):
        """
        Test the model
        :param batch: batch of images to train from
        :param batch_idx: current batch index
        :return:
        """
        x, y, img_paths = batch[0], batch[1].to(torch.int64), batch[2]

        # if ITA used
        if self.ITA:
            augmented_logits = []
            for image in x:
                # Generate augmented images (simple rotation in this case)
                augmented_images = [transforms.RandomRotation(angle)(image) for angle in range(0, 360, 45)]
                # Make predictions on augmented images
                logits_list = [self.forward(aug_img) for aug_img in augmented_images]

                # OLD CODE
                # mean_logits = torch.stack(logits_list).mean(dim=0)

                # NEW CODE
                binary_masks = []
                for logits in logits_list:
                    max_indices = torch.argmax(logits, dim=1)
                    binary_mask = torch.zeros_like(logits)
                    binary_mask[torch.arange(logits.shape[0]), max_indices] = 1
                    binary_masks.append(binary_mask)
                binary_masks = torch.stack(binary_masks, dim=0)
                index_counts = torch.sum(binary_masks, dim=0)
                max_index = torch.argmax(index_counts)

                # Create a tensor with zeros everywhere except for the maximum index
                max_logits = torch.zeros_like(index_counts)
                max_logits[torch.arange(index_counts.shape[0]), max_index] = 1

                # augmented_logits.append(mean_logits)
                augmented_logits.append(max_logits)

            logits = torch.cat(augmented_logits, dim=0)
            probabilities = logits
        else:
            # forward the images
            logits = self.forward(x)
            # calculate the probabilities
            probabilities = F.softmax(logits, dim=1)

        # calculate the loss and update accuracy
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # masks for images
        correct_mask = (preds == y) & (torch.max(probabilities, dim=1)[0] > 0.7)
        incorrect_mask = (preds != y) & (torch.max(probabilities, dim=1)[0] > 0.7)
        uncertain_mask = (torch.max(probabilities, dim=1)[0] < 0.7) & (torch.max(probabilities, dim=1)[0] > 0.3)

        # store the images according to masks
        self.correct_predictions.extend(
            [(img_paths[i], y[i], probabilities[i]) for i in range(len(y)) if correct_mask[i]])
        self.incorrect_predictions.extend(
            [(img_paths[i], y[i], probabilities[i]) for i in range(len(y)) if incorrect_mask[i]])
        self.uncertain_predictions.extend(
            [(img_paths[i], y[i], probabilities[i]) for i in range(len(y)) if uncertain_mask[i]])

        # log the results
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        """
        Called at the end of epoch, used for displaying the correct/incorrect/uncertain images
        :return: None
        """

        def display_examples(predictions, title):
            """
            Display the images
            :param predictions: images predictions
            :param title: title for plot
            :return: None
            """
            plt.figure(figsize=(15, 5))
            plt.suptitle(f'{title}. Chosen fold: {self.fold_index + 1}')
            plt.subplots_adjust(top=0.8)

            for i, (image_path, label, prob) in enumerate(predictions[:3]):
                image = Image.open(image_path)
                predicted_label = torch.argmax(prob).item()

                plt.subplot(1, 3, i + 1)
                plt.imshow(image)
                plt.title(
                    f'Actual: {label.item()}\n'
                    f'Predicted: {predicted_label}\n'
                    f'Actual probability: {prob[label]:.4f}\n'
                    f'Predicted probability: {prob[predicted_label]:.4f}')
            plt.show()

        # display each type of images
        display_examples(self.correct_predictions, 'Correctly Classified Examples')
        display_examples(self.incorrect_predictions, 'Incorrectly Classified Examples')
        display_examples(self.uncertain_predictions, 'Uncertain Predictions')

        # reset stored predictions for the next tests
        self.correct_predictions = []
        self.incorrect_predictions = []
        self.uncertain_predictions = []

    def configure_optimizers(self):
        """
        Configure the model's optimizers
        :return: None
        """
        # if constants learning rate
        if self.learning_rate:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer
        # if not use scheduler
        else:
            optimizer = torch.optim.Adam(self.parameters())
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'val_loss'
                }
            }

    def setup(self, stage=None):
        """
        Setup the model's datasets
        :param stage:
        :return: None
        """
        if stage == "fit" or stage is None:
            self.fish_train = FishDataset(annotations=self.annotations_train,
                                          img_dir=self.dataset_path,
                                          transform=self.image_transform)

            self.fish_val = FishDataset(annotations=self.annotations_val,
                                        img_dir=self.dataset_path,
                                        transform=self.image_transform)

        if stage == "test" or stage is None:
            self.fish_test = FishDataset(annotations=self.annotations_test,
                                         img_dir=self.dataset_path,
                                         transform=self.image_transform)

    def train_dataloader(self):
        return DataLoader(self.fish_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fish_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fish_test, batch_size=self.batch_size)

    def on_train_end(self):
        """
        Called when the training phase ends. Used for printing graphs of accuracy and loss.
        :return:
        """

        def print_stats(train_values, val_values, type):
            """
            Print statistics
            :param train_values: values for training set
            :param val_values: values for validation set
            :param type: Accuracy/Loss
            :return:
            """
            epochs = range(1, len(train_values) + 1)
            plt.plot(epochs, train_values, marker='o', label=f'Train {type}')
            plt.plot(epochs, val_values, marker='o', label=f'Validation {type}')
            plt.xlabel('Epoch')
            plt.ylabel(type)
            plt.title(f'{type} over epochs for fold {self.fold_index + 1}')
            plt.legend()
            plt.grid(True)
            plt.show()

        # print stats
        print_stats([v.cpu().numpy() for v in self.train_accs], [v.detach().cpu().numpy() for v in self.val_accs][1:],
                    'Accuracy')
        print_stats([v.detach().cpu().numpy() for v in self.train_losses],
                    [v.detach().cpu().numpy() for v in self.val_losses][1:], 'Loss')
