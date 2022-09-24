import torch
import torchvision.transforms as transforms
import os.path
from PIL import Image
from csv import reader


img_size = 224


# convert a single image to tensor
def read_image(path_img):
    # load image in RGB mode (png files contains additional alpha channel)
    img = Image.open(path_img).convert('RGB')

    # set up transformation to resize the image
    resize = transforms.Resize([img_size, img_size])
    img = resize(img)
    to_tensor = transforms.ToTensor()

    # apply transformation and convert to Pytorch tensor
    tensor = to_tensor(img)
    # torch.Size([3, 224, 224])

    # add another dimension at the front to get NCHW shape
    tensor = tensor.unsqueeze(0)
    return tensor


# count how many files there are in the directory
def num_files(path_img):
    return len([entry for entry in os.listdir(path_img) if os.path.isfile(os.path.join(path_img, entry))])


# make a tensor for each img and a tensor with the corresponding labels
def read_data(path_csv, path_img):
    tensor_img = torch.zeros((num_files(path_img), 3, img_size, img_size))
    tensor_label = torch.zeros((num_files(path_img),))

    # skip first line i.e. read header first and then iterate over each row of csv as a list
    with open(path_csv, 'r') as read_obj:
        csv_reader = reader(read_obj)
        i = 0
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            row = row[0].split(";")
            name = row[0]
            label = row[1]
            path = path_img + str(name)
            if os.path.exists(path):
                tensor_img[i] = read_image(path)
                tensor_label[i] = age2class20(int(label))
                #tensor_label[i] = int(label)
                i = i + 1

    return tensor_img, tensor_label


def age2class(age):
    if 0 <= age <= 4:
        return 0
    elif 5 <= age <= 9:
        return 1
    elif 10 <= age <= 14:
        return 2
    elif 15 <= age <= 20:
        return 3
    elif 21 <= age <= 26:
        return 4
    elif 27 <= age <= 35:
        return 5
    elif 36 <= age <= 43:
        return 6
    elif 44 <= age <= 50:
        return 7
    elif 51 <= age <= 62:
        return 8
    else:  # 63 <= age <= 100:
        return 9


def age2class10(age):
    if 0 <= age <= 9:
        return 0
    elif 10 <= age <= 19:
        return 1
    elif 20 <= age <= 29:
        return 2
    elif 30 <= age <= 39:
        return 3
    elif 40 <= age <= 49:
        return 4
    elif 50 <= age <= 59:
        return 5
    elif 50 <= age <= 69:
        return 6
    elif 70 <= age <= 79:
        return 7
    elif 80 <= age <= 89:
        return 8
    else:  # 63 <= age <= 100:
        return 9


def age2class20(age):
    if 0 <= age <= 4:
        return 0
    elif 5 <= age <= 9:
        return 1
    elif 10 <= age <= 14:
        return 2
    elif 15 <= age <= 19:
        return 3
    elif 20 <= age <= 24:
        return 4
    elif 25 <= age <= 29:
        return 5
    elif 30 <= age <= 34:
        return 6
    elif 35 <= age <= 39:
        return 7
    elif 40 <= age <= 44:
        return 8
    elif 45 <= age <= 49:
        return 9
    elif 50 <= age <= 54:
        return 10
    elif 55 <= age <= 59:
        return 11
    elif 60 <= age <= 64:
        return 12
    elif 65 <= age <= 69:
        return 13
    elif 70 <= age <= 74:
        return 14
    elif 75 <= age <= 79:
        return 15
    elif 80 <= age <= 84:
        return 16
    elif 85 <= age <= 89:
        return 17
    elif 90 <= age <= 94:
        return 18
    else:  # >= 95:
        return 19