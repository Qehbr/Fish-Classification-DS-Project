import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_train_test_split(test_ratio, dataset_path, split_seed=12345, get_val_data=False):
    """
    Get train test (possibly val) pd dataframes
    :param test_ratio: test (val) ratio
    :param dataset_path: path to images
    :param split_seed: random seed
    :param get_val_data: get validation data or not
    :return: train, test (val) pd dataframes
    """
    fish_list = []
    # get all images in folders
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path)]

            fish_list.append(
                pd.DataFrame({'path': [os.path.join(class_folder, img) for img in images],
                              'class': [class_folder] * len(images)}))
    fish_pd = pd.concat(fish_list, ignore_index=True)

    # split to train and test
    fish_train_val, fish_test = train_test_split(
        fish_pd,
        test_size=test_ratio,
        stratify=fish_pd['class'],
        random_state=split_seed
    )

    if get_val_data:
        # if validation get validation from train data
        fish_train, fish_val = train_test_split(
            fish_train_val,
            test_size=test_ratio,
            stratify=fish_train_val['class'],
            random_state=split_seed
        )

        # encode the labels
        label_encoder = LabelEncoder()
        fish_train['class'] = label_encoder.fit_transform(fish_train['class'])
        fish_test['class'] = label_encoder.transform(fish_test['class'])
        fish_val['class'] = label_encoder.transform(fish_val['class'])
        return fish_train, fish_val, fish_test
    else:
        # encode the labels
        label_encoder = LabelEncoder()
        fish_train_val['class'] = label_encoder.fit_transform(fish_train_val['class'])
        fish_test['class'] = label_encoder.transform(fish_test['class'])

        return fish_train_val, fish_test
