import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class CIFARDataset(Dataset):
    def __init__(self, root="data", train=True):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]

        self.images = []
        self.labels = []

        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.images.extend(dict[b'data'])
                self.labels.extend(dict[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((3, 32, 32)).astype(np.float32)
        label = self.labels[idx]
        return image/255., label

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.lables = []
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
        self.transform = transform

        data_path = os.path.join(root, "animals")

if __name__ == '__main__':
    dataset = CIFARDataset(root="./data", train=True)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    for images, labels in train_dataloader:
        print(images.shape)
        print(labels)