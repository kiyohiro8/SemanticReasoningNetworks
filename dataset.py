import os

import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImgTxtDataset(Dataset):
    def __init__(self, image_dir, csv_path, max_char_length, char_encoder=None, is_train=True):
        self.image_dir = image_dir
        self.image_file_list = []
        self.text_list = []
        with open(csv_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            l_list = line.rstrip().split(",")
            image_file = l_list[0]
            text = l_list[1]
            self.image_file_list.append(image_file)
            self.text_list.append(text)
        if char_encoder is not None:
            self.char_encoder = char_encoder
        else:
            char_list = get_char_list(self.text_list)
            self.char_encoder = LabelEncoder()
            self.char_encoder.fit(char_list)
        self.stop_char_index = len(self.char_encoder.classes_)
        self.max_char_length = max_char_length
        if is_train:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                Pad(),
                transforms.RandomAffine(degrees=5, scale=(0.8, 1.2), shear=10, resample=PIL.Image.BILINEAR),
                transforms.Resize((128, 32)),
                transforms.ToTensor()
            ])
        self.is_train = is_train

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_file_list[index])
        image = Image.open(image_path)
        if self.is_train:
            image = self.transform(image)
        
        char_list = list(self.text_list[index])
        enc_char_list = self.char_encoder.transform(char_list)
        len_char = len(enc_char_list)
        enc_char_tensor = np.array(enc_char_list)
        text_tensor = torch.LongTensor(self.max_char_length).fill_(self.stop_char_index)
        text_tensor[:len_char] = enc_char_tensor

        return image, text_tensor

    def __len__(self):
        return len(self.image_file_list)


class Pad(object):
    def __init__(self, min_w=128, min_h=32):
        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, image):
        w, h = image.size
        if w >= self.min_w and h >= self.min_h:
            return image
        new_w = max(w, self.min_w)
        new_h = max(h, self.min_h)
        padded = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
        padded.paste(image, (0, 0))
        return padded

def get_char_list(text_list):
    char_set = set([])
    for text in text_list:
        char_list = list(text)
        char_set = char_set | set(char_list)
    result_char_list = list(char_set)
    return sort(result_char_list)