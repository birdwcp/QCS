import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def config(dataset):
    if dataset == 'RAF-DB':
        train_root = '../datas/RAF-DB/basic'
        test_root = '../datas/RAF-DB/basic'
        train_pd = pd.read_csv("../datas/RAF-DB/raf_db_basic_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("../datas/RAF-DB/raf_db_basic_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 7
    if dataset == 'AffectNet-7':
        train_root = '../datas/AffectNet/Manually_trainval_croped'
        test_root = '../datas/AffectNet/Manually_trainval_croped'
        train_pd = pd.read_csv("../datas/AffectNet/training-cls7-v1.txt", sep=" ", header=None,
                               names=['ImageName', 'label'], engine='python')
        test_pd = pd.read_csv("../datas/AffectNet/validation-cls7.txt", sep=" ", header=None,
                              names=['ImageName', 'label'], engine='python')
        cls_num = 7

    if dataset == 'AffectNet-8':
        train_root = '../datas/AffectNet/Manually_trainval_croped'
        test_root = '../datas/AffectNet/Manually_trainval_croped'
        train_pd = pd.read_csv("../datas/AffectNet/training-cls8-v1.txt", sep=" ", header=None,
                               names=['ImageName', 'label'], engine='python')
        test_pd = pd.read_csv("../datas/AffectNet/validation-cls8.txt", sep=" ", header=None,
                              names=['ImageName', 'label'], engine='python')
        cls_num = 8

    if dataset == 'FERPlus':
        train_root = '../datas/FERPlus/img'
        test_root = '../datas/FERPlus/img'
        train_pd = pd.read_csv("../datas/FERPlus/ferplus_training-v2.txt", sep=" ", header=None,
                               names=['ImageName', 'label'], engine='python')
        test_pd = pd.read_csv("../datas/FERPlus/ferplus_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'], engine='python')
        cls_num = 8

    if dataset == 'FERPlus+RAF-DB':
        train_root = '../datas/'
        test_root = '../datas/'
        train_pd = pd.read_csv("../datas/FERPlus+RAF-DB/ferplus_raf_db_training.txt", sep=" ", header=None,
                               names=['ImageName', 'label'], engine='python')
        test_pd = pd.read_csv("../datas/FERPlus+RAF-DB/ferplus_raf_db_test-cls7.txt", sep=" ", header=None,
                              names=['ImageName', 'label'], engine='python')
        cls_num = 8

    return train_root, test_root, train_pd, test_pd, cls_num


class Dataset(Dataset):
    def __init__(self, root_dir, pd_file, train=False, transform=None, num_positive=1, num_negative=1):
        self.root_dir = root_dir
        self.pd_file = pd_file
        self.image_names = pd_file['ImageName'].tolist()
        self.labels = pd_file['label'].tolist()

        self.train = train
        self.transform = transform

        self.num_positive = num_positive


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.image_names[item])
        image = self.pil_loader(img_path)
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        if self.train:
            positive_image = self.fetch_positive(self.num_positive, label, self.image_names[item])
            return image, positive_image, label
        return image, label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def fetch_positive(self, num_positive, label, path):
        other_img_info = self.pd_file[(self.pd_file.label == label) & (self.pd_file.ImageName != path)]
        other_img_info = other_img_info.sample(min(num_positive, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.root_dir, e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        positive_img = [self.transform(img) for img in other_img]
        return positive_img

    def fetch_negative(self, num_negative, label, path):
        other_img_info = self.pd_file[(self.pd_file.label != label) & (self.pd_file.ImageName != path)]
        other_img_info = other_img_info.sample(min(num_negative, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.root_dir, e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        negative_img = [self.transform(img) for img in other_img]

        for e in other_img_info:
            neg_label = e['label']

        return negative_img, neg_label

    def get_labels(self):
        return self.labels


def collate_fn(batch):
    imgs = []
    positive_imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        positive_imgs.extend(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(positive_imgs, 0), labels
