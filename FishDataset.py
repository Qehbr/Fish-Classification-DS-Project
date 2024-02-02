import os
from PIL import Image
from torch.utils.data import Dataset


class FishDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, target_transform=None, get_path=True):
        """
        FishDataset constructor
        :param annotations: pd dataframe with (image_path / label encoded class)
        :param img_dir: directory with images
        :param transform: transform for images
        :param target_transform: transform for labels
        :param get_path: return path to image in getitem or not
        """
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.get_path = get_path

    def __len__(self):
        """
        Returns the length of dataset
        :return: length of dataset
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Returns the image and label from dataset
        :param idx:
        :return: image and label
        """
        # open images
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        # transform image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # if return path of the image
        if self.get_path:
            return image, label, img_path
        else:
            return image, label
