# import the necessary libraries
import os

import cv2
import numpy as np
import torch

from kakuro_csp_solver import KakuroCustomGame, KakuroUI
from utils.cell_details import get_cell_status, get_cell_hint
from utils.constants import Constants
from utils.generate_puzzle_blocks import generate_puzzle_blocks


DATA_PATH = '../data/puzzles/'
MAX_HINTS = 26
PUZZLE_SIZE = 9
prediction_to_num = {
    108: 1,
    1073: 6,
    115: 8
} # this dictionary will store the numbers which are predicted with a different ascii value.

def read_puzzle_from_image(img: np.ndarray) -> np.ndarray:
    """
    A function to read the kakuro puzzle from image by parsing every individual cell.
    :param img: the image which contains the puzzle.
    :type img: `numpy.ndarray`
    :return: the puzzle read as an array of strings.
    :returns: `numpy.ndarray`
    """
    puzzle_string = ''
    row_count, column_count = 0, 0
    curr_row_index, curr_column_index = 4, 4
    global prediction_to_num

    # code to access every box in the puzzle
    while row_count < Constants.ROWS:
        column_count = 0
        curr_column_index = 4

        while column_count < Constants.COLUMNS:
            curr_cell = img[
                curr_row_index: curr_row_index + Constants.CELL_SIDE_LENGTH,
                curr_column_index: curr_column_index + Constants.CELL_SIDE_LENGTH
                ]
            cell_status = get_cell_status(curr_cell)
            if cell_status == 'empty':
                puzzle_string += ' ,'
            elif cell_status == 'filled':
                puzzle_string += ' \ ,'
            else:
                left_hint, prediction_to_num = get_cell_hint(curr_cell, prediction_to_num)
                right_hint, prediction_to_num = get_cell_hint(curr_cell, prediction_to_num, triangle_side='right')
                puzzle_string += '{}\{},'.format(left_hint if left_hint > 0 else ' ', right_hint if right_hint > 0 else ' ')

            curr_column_index += Constants.CELL_SIDE_LENGTH + Constants.CELL_TO_CELL_GAP
            column_count += 1

        curr_row_index += Constants.CELL_SIDE_LENGTH + Constants.CELL_TO_CELL_GAP
        puzzle_string = puzzle_string[:-1] + '\n'
        row_count += 1

    puzzle_matrix = np.array([np.array(row.split(',')) for row in puzzle_string[:-1].split('\n')])
    puzzle = KakuroCustomGame(puzzle_string)
    kak = KakuroUI(puzzle)
    kak.solve()

    # substitute the vacant spots with the actual numbers
    for pos_x, pos_y, num in kak.game.data_filled:
        puzzle_matrix[pos_x][pos_y] = str(num)

    return puzzle_matrix

final_all_blocks = np.empty((0, MAX_HINTS, PUZZLE_SIZE))

# iterate over every puzzle and generate the puzzle matrix
for puzzle_difficulty in os.listdir(DATA_PATH):
    all_blocks = np.array([])
    print(f'Processing puzzles with difficulty: {puzzle_difficulty}')
    puzzle_images = os.listdir(os.path.join(DATA_PATH, puzzle_difficulty))
    for idx, puzzle in enumerate(puzzle_images):
        print(f'{idx} puzzles processed.', end=' ')
        print(os.path.join(DATA_PATH, puzzle_difficulty, puzzle))
        img = cv2.imread(
            os.path.join(DATA_PATH, puzzle_difficulty, puzzle),
            cv2.IMREAD_GRAYSCALE
        )
        puzzle_matrix = read_puzzle_from_image(img)
        blocks = generate_puzzle_blocks(puzzle=puzzle_matrix)
        # if blocks is None, then it means that the solver couldn't find
        # a valid solution
        if blocks is None:
            print('Invalid puzzle. Skipping.')
            continue

        if blocks.shape[0] != MAX_HINTS:
            filler_block = np.array([
                [-1 for _ in range(PUZZLE_SIZE)] for __ in range(MAX_HINTS - len(blocks))
            ])
            blocks = np.vstack((blocks, filler_block))
        if len(all_blocks) == 0:
            all_blocks = np.append(all_blocks, blocks)
            all_blocks = np.reshape(all_blocks, (1, *blocks.shape))
        else:
            blocks = np.reshape(blocks, (1, *blocks.shape))
            all_blocks = np.vstack((all_blocks, blocks))
    final_all_blocks = np.vstack((final_all_blocks, all_blocks))

# save the vectors generated
if not os.path.exists('../data/vectors'):
    os.mkdir('../data/vectors')

tensors = torch.from_numpy(final_all_blocks)
torch.save(tensors, '../data/vectors/basic_tensors.pt')

print(prediction_to_num)