# import the necessary libraries
import argparse
import os
import sys
from time import time
sys.path.append('/home/pramod/MS/SEM-1/NSR/SATNet/satnet')

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

import satnet

DATA_PATH = './data/vectors/'
OUTPUT_TENSOR = 'basic_tensors.pt'
INPUT_TENSOR = 'X.pt'
IS_INPUT_TENSOR = 'is_input.pt'
NO_OF_EPOCHS = 150
LEARNING_RATE = 2e-3
BATCH_SIZE = 50
BLOCKS_PER_PUZZLE = 26
BLOCK_LENGTH = 8
CLASSES = 11
RANDOM_STATE = np.random.randint(1, 101)

torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class KakuroSolver(nn.Module):
    """
    A class to define the SATNet model for solving kakuro puzzles
    """

    def __init__(self, n, m, aux):
        super(KakuroSolver, self).__init__()
        self.sat = satnet.SATNet(n, m, aux)

    def forward(self, batch, mask):
        out = self.sat(batch, mask)
        return out


def get_pred_accuracy(preds, labels, is_input):
    """
    TODO: Update the documentation
    """
    pred_reshape = torch.reshape(preds, (-1, BLOCKS_PER_PUZZLE, BLOCK_LENGTH, CLASSES)).argmax(dim=3)
    is_input_reshape, _ = torch.reshape(is_input, (-1, BLOCKS_PER_PUZZLE, BLOCK_LENGTH, CLASSES)).max(dim=3)
    labels_reshape = torch.reshape(labels, (-1, BLOCKS_PER_PUZZLE, BLOCK_LENGTH, CLASSES)).argmax(dim=3)

    # filter out the places where is_input is equal to 1
    pred_reshape = pred_reshape[is_input_reshape == 0]
    labels_reshape = labels_reshape[is_input_reshape == 0]
    comparison = (pred_reshape == labels_reshape)

    return torch.numel(comparison[comparison == True]) / len(comparison)


def get_device(use_gpu=True):
    """
    A function to initialise the device which will be used for training.
    """
    if torch.cuda.is_available() and use_gpu:
        print(f'GPU available: {torch.cuda.get_device_name()}')
        device = torch.device('cuda')
    else:
        print('No GPU available or forbidden from usage. Using CPU')
        device = torch.device('cpu')
    return device


def transform(data):
    """
    1 - 9 take their respective (numbers - 1)th index.
    -1 takes index 10 and any hint number takes index 9.
    :param data: a 1-D array.
    """
    encoded = np.zeros((*data.shape, 11))
    for p_idx, puzzle in enumerate(data):
        for r_idx, row in enumerate(puzzle):
            for n_idx, num in enumerate(row):
                if n_idx == 0:
                    encoded[p_idx][r_idx][n_idx][9] = 1
                elif num == -1:
                    encoded[p_idx][r_idx][n_idx][10] = 1
                elif num == 0:
                    continue
                else:
                    encoded[p_idx][r_idx][n_idx][num - 1] = 1
    return encoded.reshape((data.shape[0], data.shape[1] * data.shape[2] * 11))


def transform_is_input(data):
    encoded = np.zeros((*data.shape, 11))
    for p_idx, puzzle in enumerate(data):
        for r_idx, row in enumerate(puzzle):
            for n_idx, num in enumerate(row):
                if num == 1:
                    for ohe_idx in range(11):
                        encoded[p_idx][r_idx][n_idx][ohe_idx] = 1
    return encoded.reshape((data.shape[0], data.shape[1] * data.shape[2] * 11))


def run(epoch: int, model: KakuroSolver, optimizer:torch.optim, dataset: torch.utils.data.Dataset, batchSz: int, device: torch.device, to_train: bool) -> float:
    """
    :return: the loss observed in the current epoch.
    """

    loss_final = 0
    acc_final = 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSz, drop_last=True)

    for idx, (data, is_input, label) in enumerate(loader):
        # transfer the data to the designated device.
        b_data = data.to(device)
        b_is_input = is_input.to(device)
        b_label = label.to(device)

        if to_train: optimizer.zero_grad()
        preds = model(b_data.contiguous(), b_is_input.contiguous())
        loss = nn.functional.binary_cross_entropy(preds, b_label)

        if to_train:
            loss.backward()
            optimizer.step()

        loss_final += loss.item()
        acc_final += get_pred_accuracy(preds, b_label, b_is_input)

    loss_final = loss_final/len(loader)
    acc_final = acc_final / len(loader)

    print('Epoch {}: RESULTS: Average loss: {:.4f}. Average accuracy: {:.4f}'.format(epoch, loss_final, acc_final))

    torch.cuda.empty_cache()
    return loss_final, acc_final


def train(epoch, model, optimizer, dataset, batchSz, device):
    return run(epoch, model, optimizer, dataset, batchSz, device, to_train=True)


@torch.no_grad()
def test(epoch, model, optimizer, dataset, batchSz, device):
    return run(epoch, model, optimizer, dataset, batchSz, device, to_train=False)


def reverse_ohe(data_point, shape, func, dim):
    """
    For debugging purpose.
    """
    data_point = torch.reshape(data_point, shape)
    data_point = func(data_point, dim=dim)
    print(data_point)


losses = []
accuracies = []

# load the input vectors
X = torch.load(os.path.join(DATA_PATH, INPUT_TENSOR))
y = torch.load(os.path.join(DATA_PATH, OUTPUT_TENSOR))
is_input = torch.load(os.path.join(DATA_PATH, IS_INPUT_TENSOR))

X = X[:, :, :-1]
init_shape = X.shape
X = transform(X)
X = torch.from_numpy(X).type(torch.float32)

y = y[:, :, :-1].type(torch.int32)
y = torch.from_numpy(transform(y)).type(torch.float32)

is_input = is_input[:, :, :-1]
is_input = torch.from_numpy(transform_is_input(is_input)).type(torch.int32)

X_train, X_test, y_train, y_test, is_input_train, is_input_test =\
    train_test_split(X, y, is_input, test_size=0.2)

train_dataset = torch.utils.data.TensorDataset(X, is_input, y)

solver = KakuroSolver(X.shape[1], 600, 300)

# get ready to train
# fetch the device which will be used to train
use_gpu = False if len(sys.argv) > 1 and sys.argv[1] == '0' else True
device = get_device(use_gpu)

# move the model to the `device`
solver.to(device)

print('Starting the training process...')
optimizer = torch.optim.Adam(solver.parameters(), lr=LEARNING_RATE)
for epoch in range(1, NO_OF_EPOCHS + 1):
    epoch_loss, epoch_accuracy = train(
        epoch=epoch,
        model=solver,
        optimizer=optimizer,
        dataset=train_dataset,
        batchSz=BATCH_SIZE,
        device=device
    )
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

# create a folder to save the models
if not os.path.exists('./data/models'):
    os.mkdir('./data/models')

# create a folder to save the loss plot
if not os.path.exists('./data/plots'):
    os.mkdir('./data/plots')

timestamp = int(time())

# create a folder to save the loss plots
torch.save(solver.state_dict(), f'./data/models/satnet_{timestamp}.pt')

# save the plot
plt.plot(range(1, len(losses) + 1), losses, label='losses')
plt.plot(range(1, len(accuracies) + 1), accuracies, label='accuracies')
plt.legend()
plt.savefig(f'./data/plots/plot_{timestamp}.png')