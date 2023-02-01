from torchvision.transforms import transforms
import os
import torch
from PIL import Image

import random
from torch.utils.data import Dataset


class Clothin1m(Dataset):
    def __init__(self, root, mode, subset=None):
        self.root = root
        self.anno_dir = os.path.join(self.root, "annotations")
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.ConvertImageDtype(torch.float)
            ]
        )
        self.mode = mode

        self.imgs = self.load_images()
        if subset is not None:
            self.imgs = random.choices(self.imgs, k=int(self.__len__() * subset))
        self.labels = self.get_label_map()

    def load_images(self):
        path = os.path.join(self.anno_dir, 'clean_train_key_list.txt') if self.mode == 'train' else os.path.join(
            self.anno_dir, 'clean_test_key_list.txt') if self.mode == 'test' else os.path.join(self.anno_dir,
                                                                                               'clean_val_key_list.txt') if self.mode == 'val' else os.path.join(
            self.anno_dir, 'noisy_train_key_list.txt') if self.mode == 'noisy_train' else None

        if path is None: raise ValueError('Mode not supported!')

        with open(path) as f:
            return f.read().splitlines()

    def get_label_map(self):
        path = os.path.join(self.anno_dir, 'clean_label_kv.txt') if self.mode != 'noisy_train' else os.path.join(
            self.anno_dir, 'noisy_label_kv.txt')
        with open(path) as f:
            content = f.read().splitlines()
            content = [line.split(' ') for line in content]
            label_map = {}
            for line in content:
                label_map[line[0]] = int(line[1])
            return label_map

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[img_path]

        image = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.imgs)
