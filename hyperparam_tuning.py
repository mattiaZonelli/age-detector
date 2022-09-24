import os
import os.path as osp
import random
from argparse import ArgumentParser
from functools import partial
import torch.nn as nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, sampler
from torchvision.models import vgg16, VGG16_Weights
import torch.optim as optim
from tqdm import tqdm
import torch
from age_dataset import *
import time
import matplotlib.pyplot as plt

NUM_CLASSES = 20
NUM_SAMPLES = 256
VERBOSE = False
PATH = "./age_det_20classes.pt"


def get_dsets(config):
    trset = CustomImageDataset(annotations_file=config["csv_dir"], img_dir=config["train_img"])
    vlset = CustomImageDataset(annotations_file=config["csv_dir"], img_dir=config["vali_img"])

    return trset, vlset


def load_data(config):
    trset, vlset = get_dsets(config)


    trload = DataLoader(trset, batch_size=config["batch_size"], num_workers=1, shuffle=True)
    vlload = DataLoader(vlset, 1, num_workers=1)
    return trload, vlload


# get the model to use
def get_model():
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    output_ftrs = NUM_CLASSES
    # change the fully connected layer with one for our purpose
    model.classifier[6] = nn.Linear(in_features=4096, out_features=output_ftrs)
    return model


def train(config):
    model = get_model()

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
    print(device, "\n")
    model.to(device)

    #set criteria to compute loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          weight_decay=config["weight_decay"],
                          momentum=config["momentum"])
    accuracy = Accuracy()

    trload, vlload = load_data(config)
    tr_losses = torch.ones(config['epoch'])
    val_losses = torch.ones(config['epoch'])
    for epoch in range(config['epoch']):
        # training
        running_loss = 0.
        epoch_steps = 0
        pbar = _get_pbar(trload, "TRAIN")
        start_time = time.time()
        for iter_, data in enumerate(pbar, 1):

            loss = iteration(data, optimizer, criterion, model, device)  # for age
            running_loss += loss.item()
            epoch_steps += 1

            if VERBOSE:
                pbar.set_description(f"TRAIN - epoch: {epoch} - LOSS: {running_loss / epoch_steps:.4f}")

        tr_losses[epoch] = running_loss / epoch_steps

        correct, total, val_loss = validation(model, config, device)
        val_losses[epoch] = val_loss / total
        # print("\nVAL - epoch:", epoch, " | loss: ", val_loss)

    model.train()
    accuracy.reset()
    fileName = r'age_det_101classes-loss.png'
    fig, ax = plt.subplots(1)
    plt.plot(range(config['epoch']), tr_losses, label='Train Loss')
    plt.plot(range(config['epoch']), val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    fig.savefig(fileName, format='png')
    plt.close(fig)
    print("Finished Training")
    return model


def _get_pbar(loader, desc):
    if VERBOSE:
        return tqdm(loader, desc=desc)
    return loader


# iteration step for training on rotation
def iteration(data, optimizer, criterion, model, device):
    optimizer.zero_grad()

    image, label = data

    output = model(x=image.to(device))

    label = label.type(torch.LongTensor)
    before = list(model.parameters())

    loss = criterion(output, label.to(device))
    loss.backward()
    optimizer.step()  # update weights

    after = model.parameters()
    assert before != after
    return loss


# validation
def validation(model, config, device):
    accuracy = Accuracy(num_classes=NUM_CLASSES, top_k=1)
    criterion = nn.CrossEntropyLoss()

    _, vlload = load_data(config)
    pbar = _get_pbar(vlload, "VAL")
    model.eval()
    total = 0
    correct = 0
    loss = 0.
    with torch.no_grad():
        for data in pbar:
            image, label = data

            output = model(x=image.to(device))

            label = label.type(torch.LongTensor)

            loss += criterion(output, label.to(device)).item()

            m = nn.Softmax(dim=1)
            output = m(output)

            dacc = accuracy(output.to('cpu'), label.to('cpu'))

            total += output.shape[0]
            #correct += (output.squeeze().argmax() == label).sum().item()
            correct += dacc.item()

            if VERBOSE:
                pbar.set_description(
                    f"VAL - ACC.: {correct / total:.4f} | pred: {output.squeeze().argmax()}, label: {label.item()} - LOSS: {loss / total}")
            else:
                print(f"VAL - ACC.: {correct / total:.4f} | pred: {output.squeeze().argmax()}, label: {label.item()}")
                plt.imshow(image.squeeze().permute(1, 2, 0))
                plt.show()

    model.train()
    accuracy.reset()

    return correct, total, loss


if __name__ == "__main__":
    # set_seed(args.seed)
    torch.cuda.empty_cache()

    '''model_config = {"batch_size": 16,
                    "csv_dir": "./Train.csv",
                    "train_img": "./train/",
                    "vali_img": "./validation/",
                    "lr": 0.0001,
                    "weight_decay": 0.0005,
                    "momentum": 0.9,
                    "epoch": 25
                    }
    age_det = train(model_config)
    torch.save(age_det.state_dict(), PATH)'''

    # to do test
    test_config = {"batch_size": 1,
                   "csv_dir": "/Train.csv",
                   "train_img": "/train/",
                   "vali_img": "/validation/",
                   }
    my_model = get_model()
    my_model.load_state_dict(torch.load(PATH, map_location='cpu'))
    validation(my_model, config=test_config, device='cpu')
    print("FINISH test phase")
