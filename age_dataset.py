import pandas as pd
from torch.utils.data import Dataset
from utils import *


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_dir = img_dir
        self.tensor_img, self.tensor_label = read_data(annotations_file, img_dir)

    def __len__(self):
        return num_files(self.img_dir)

    def __getitem__(self, idx):
        return self.tensor_img[idx], self.tensor_label[idx]
