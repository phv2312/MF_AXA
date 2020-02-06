import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import imutils
import os, glob


def imshow(image, is_cv2=False):
    if is_cv2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def prepare_data(data_v, data_h):
    paths_v = glob.glob(os.path.join(data_v, '*'))
    paths_h = glob.glob(os.path.join(data_h, '*'))
    for path_h in paths_h:
        image_name = os.path.basename(path_h)[:-4]
        image = cv2.imread(path_h)
        image_1 = imutils.rotate_bound(image, 180)
        # image_2 = imutils.rotate_bound(image, -90)
        cv2.imwrite(os.path.join(data_h, image_name + '_180.png'), image_1)
        # cv2.imwrite(os.path.join(data_v, image_name + '_270.png'), image_2)


class FormDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.imgs = []
        self.labels1 = []
        self.labels2 = []

        # Get images
        single_imgs = glob.glob(os.path.join(data_root, 'single', '*'))
        v_imgs = glob.glob(os.path.join(data_root, 'multiple/vertical', '*'))
        h_imgs = glob.glob(os.path.join(data_root, 'multiple/horizontal', '*'))

        self.imgs.extend(single_imgs)
        self.imgs.extend(v_imgs)
        self.imgs.extend(h_imgs)

        # Get labels
        single_labels = [0] * len(single_imgs)
        multi_labels = [1] * (len(v_imgs) + len(h_imgs))
        self.labels1.extend(single_labels)
        self.labels1.extend(multi_labels)

        single_labels = [2] * len(single_imgs)
        v_labels = [0] * len(v_imgs)
        h_labels = [1] * len(h_imgs)
        self.labels2.extend(single_labels)
        self.labels2.extend(v_labels)
        self.labels2.extend(h_labels)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        tensor_img = transforms.ToTensor()(image)
        tensor_label1 = torch.tensor(label1, dtype=torch.float32)
        tensor_label2 = torch.tensor(label2, dtype=torch.float32)
        return tensor_img, tensor_label1, tensor_label2

    def __len__(self):
        return len(self.imgs)


class FormLoader(object):
    def __init__(self, data_root, batch_size, shuffle=True):
        self.data_root = data_root
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset = FormDataset(data_root)


    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == '__main__':
    # dataset = FormDataset('/home/dotieuthien/Documents/AXA/MF_AXA/mf_axa/clf/dataset')
    # dataset.__getitem__(350)
    dataloader = FormLoader('/home/dotieuthien/Documents/AXA/MF_AXA/mf_axa/clf/dataset', 1, True)
    dataloader = dataloader.loader()
    for _, (data, label) in enumerate(dataloader):
        print(data.size())
        print(label)
