import os

import numpy as np
import torch

FOLDER_PATH = '../data/vectors'
TENSOR_FILE_NAME = 'basic_tensors.pt'
# TODO: See if the numbers below can be read from a config file of sorts?
BLOCKS_PER_PUZZLE = 26
BLOCK_LENGTH = 9
tensors = torch.load(os.path.join(FOLDER_PATH, TENSOR_FILE_NAME))
tensor_array = tensors.numpy()

# for every puzzle matrix, generate the is_input vector and the input vector.
# convert the input numbers to zero.
is_input = np.empty((0, BLOCKS_PER_PUZZLE, BLOCK_LENGTH))
X = np.empty((0, BLOCKS_PER_PUZZLE, BLOCK_LENGTH))
for puzzle in tensors:
    curr_puzzle_is_input = np.empty((0, BLOCK_LENGTH))
    curr_puzzle_X = np.empty((0, BLOCK_LENGTH))
    for block in puzzle:
        curr_block_X = np.array([])
        curr_block_is_input = np.array([])
        for idx, num in enumerate(block):
            if idx == 0 or num == -1:
                curr_block_X = np.append(curr_block_X, num)
                curr_block_is_input = np.append(curr_block_is_input, 1)
            else:
                curr_block_X = np.append(curr_block_X, 0)
                curr_block_is_input = np.append(curr_block_is_input, 0)

        curr_puzzle_X = np.vstack((curr_puzzle_X, np.reshape(curr_block_X, (1, BLOCK_LENGTH))))
        curr_puzzle_is_input = np.vstack((
            curr_puzzle_is_input, np.reshape(curr_block_is_input, (1, BLOCK_LENGTH))
        ))

    X = np.vstack((X, np.reshape(curr_puzzle_X, (1, BLOCKS_PER_PUZZLE, BLOCK_LENGTH))))
    is_input = np.vstack((is_input, np.reshape(curr_puzzle_is_input, (1, BLOCKS_PER_PUZZLE, BLOCK_LENGTH))))

is_input = torch.from_numpy(is_input)
X = torch.from_numpy(X)

torch.save(is_input, '../data/vectors/is_input.pt')
torch.save(X, '../data/vectors/X.pt')