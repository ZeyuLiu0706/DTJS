#Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import scipy.misc as m
from torchvision import datasets, transforms


class MultiMnistMooDataset(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, multi=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.multi = multi

        if self.train:
            self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                os.path.join(self.root, self.multi_training_file))
        else:
            self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                os.path.join(self.root, self.multi_test_file))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.multi:
            if self.train:
                img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
            else:
                img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.multi:
            tar = np.array([target_l, target_r])
            return img, torch.from_numpy(tar)
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path, extension):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        multi_labels_l = np.zeros((1*length),dtype=np.long)
        multi_labels_r = np.zeros((1*length),dtype=np.long)
        for im_id in range(length):
            for rim in range(1):
                multi_labels_l[1*im_id+rim] = parsed[im_id]
                multi_labels_r[1*im_id+rim] = parsed[extension[1*im_id+rim]] 
        return torch.from_numpy(parsed).view(length).long(), torch.from_numpy(multi_labels_l).view(length*1).long(), torch.from_numpy(multi_labels_r).view(length*1).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(length, num_rows, num_cols)
        multi_length = length * 1
        multi_data = np.zeros((1*length, num_rows, num_cols))
        extension = np.zeros(1*length, dtype=np.int32)
        for left in range(length):
            chosen_ones = np.random.permutation(length)[:1]
            extension[left*1:(left+1)*1] = chosen_ones
            for j, right in enumerate(chosen_ones):
                lim = pv[left,:,:]
                rim = pv[right,:,:]
                new_im = np.zeros((36,36))
                new_im[0:28,0:28] = lim
                new_im[6:34,6:34] = rim
                new_im[6:28,6:28] = np.maximum(lim[6:28,6:28], rim[0:22,0:22])
                # multi_data_im =  m.imresize(new_im, (28, 28), interp='nearest')
                multi_data_im = np.array(Image.fromarray(new_im).resize(size=(28, 28)))
                multi_data[left*1 + j,:,:] = multi_data_im
        return torch.from_numpy(parsed).view(length, num_rows, num_cols), torch.from_numpy(multi_data).view(length,num_rows, num_cols), extension



