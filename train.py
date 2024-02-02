import torch
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import lightning as L
import utils
from CNN_Fish import CNN_Fish

torch.set_float32_matmul_precision('medium')


def train(fish_train, fish_test, epochs, batch_size, learning_rate, dataset_path, num_splits, num_classes, augmentation=False,
          normalization=False, ITA=False, split_seed=12345):
    """
    Train a CNN on the fish dataset
    :param fish_train: pd dataframe with train data (image_path / label encoded class)
    :param fish_test: pd dataframe with test data (image_path / label encoded class)
    :param epochs: number of epochs
    :param batch_size: batch size to retrieve and train the images
    :param learning_rate: learning rate to use
    :param dataset_path: path to images
    :param num_splits: number of splits to use in K fold cross validation
    :param num_classes: number of classes in dataset
    :param augmentation: use augmentation or not
    :param normalization: use normalization or not
    :param ITA: use Inference Time Augmentation or not
    :param split_seed: random seed
    :return:
    """
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=split_seed)
    trainers = []
    # for each fold
    for fold_index, (train_idx, val_idx) in enumerate(kf.split(fish_train, fish_train['class'])):

        # get fold train and validation parts
        fold_train = fish_train.iloc[train_idx]
        fold_val = fish_train.iloc[val_idx]
        print(f"Fold {fold_index + 1}: Train Size - {len(fold_train)}, Val Size - {len(fold_val)}")

        logger = TensorBoardLogger("logs", name=f"original_fold_{fold_index}")
        model = CNN_Fish(fold_train, fold_val, fish_test, dataset_path, fold_index, num_classes=num_classes, batch_size=batch_size,
                         learning_rate=learning_rate, augmentation=augmentation, normalization=normalization, ITA=ITA)
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=epochs,
            log_every_n_steps=batch_size,
            logger=logger
        )

        # train the model
        trainer.fit(model)
        trainers.append((trainer, trainer.model.val_accs[-1], trainer.model.val_losses[-1]))

    # test the best trainer
    best_trainer = max(trainers, key=lambda x: x[1])[0]
    best_trainer.test()


if __name__ == '__main__':
    augmentation = True
    normalization = True
    AIT = True
    dataset_path = 'NA_Fish_Dataset'
    fish_train, fish_test = utils.get_train_test_split(0.2, dataset_path)
    train(fish_train, fish_test, epochs=50, batch_size=8, learning_rate=0.0001, dataset_path=dataset_path, num_splits=5,
          num_classes=9,
          augmentation=augmentation, normalization=normalization, AIT=AIT)
