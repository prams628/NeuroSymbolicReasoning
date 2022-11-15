import cv2
import numpy as np
import pytesseract

from utils.constants import Constants


def _get_num(cell, predictions, prediction_to_num):
    """
    This function will parse the inpu
    """
    curr_num = 0
    for idx, num in enumerate(predictions):
        actual_num = num
        if (num < 49 or num > 58) and num not in prediction_to_num:
            print(f'Enter the number at index {idx}:', end=' ')
            cv2.imshow('cell', cell)
            cv2.waitKey(1000)
            actual_num = int(input())
            prediction_to_num[num] = [actual_num]

        curr_num = curr_num * 10 + actual_num
    return curr_num


def get_cell_status(cell):
    """
    This function verifies if the current cell that is passed is empty, filled,
    or hint. The cell is said to be empty if all the pixels of the cell are white.
    The cell is said to be filled if the cell is covered in black.
    :param cell: the square grid from an image.
    :returns: a string explaining the status of the cell. This function will
        return 'empty' if the cell is
    """
    rows, columns = cell.shape
    total_pixels = rows * columns
    black_pixel_count = 0
    for x in range(rows):
        for y in range(columns):
            if cell[x][y] == 0:
                black_pixel_count += 1

    # this variable stores the percentage of black cells.
    req_per = black_pixel_count / total_pixels

    # custom threshold set to 1% for a cell to be empty
    if req_per < 0.01:
        return 'empty'
    if req_per >= 0.99:
        return 'filled'
    return 'hint'


def _get_mask(triangle_side='left'):
    """
    A helper function to help generate a mask to focus on either the right or the left triangle.
    More details on the right and left triangle can be found in the function description of
    `get_cell_hint` in the same module.
    :param triangle_side: This param indicates which side of the square is being masked out.
    :type triangle_side: str (can hold only 'left' or 'right'; defaults to 'left')
    :return: the mask generated.
    :returns: numpy.ndarray
    """
    if triangle_side not in ['left', 'right']:
        raise AssertionError(f'Triangle side parameter [{triangle_side}] not one of ["left", "right"]')
    mask = np.zeros((42, 42, 3))
    white_pixel = np.array([255, 255, 255])
    for i in range(42):
        triangle_side_range = range(i - 1, -1, -1) if triangle_side == 'left' else range(i, 42)
        for j in triangle_side_range:
            mask[i][j] = white_pixel.copy()
    return mask


def get_cell_hint(cell, prediction_to_num, triangle_side='left', inverted=False):
    """
    #TODO: Fix the documentation.
    This function helps in fetching the hints in a hint cell. Point to note: The right diagonal of a square splits it
    into two right-angled triangles. One to the right and one to the left.
    :param cell: The cell under focus.
    :type cell: ndarray
    :param triangle_side: This param indicates which side of the square is being masked out.
    :type triangle_side: str (can hold only 'left' or 'right'; defaults to 'left')
    :return: the hint
    :returns: TBD # TODO: Fix the return type here.
    """
    if not inverted:
        cell = cv2.bitwise_not(cell)
    mask_side = 'right' if triangle_side == 'left' else 'left'
    mask = _get_mask(triangle_side=mask_side).astype(np.uint8)

    # erase the diagonal line in the image.
    for idx in range(cell.shape[0]):
        cell[idx][idx] = 255
        if idx + 1 < cell.shape[0]:
            cell[idx][idx + 1] = 255
            cell[idx + 1][idx] = 255

    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
    cell = cv2.bitwise_or(cell, mask)

    # now, run the OCR
    data = pytesseract.image_to_string(cell, lang='osd', config='--psm 13')
    predicted_num = 0
    for idx, ele in enumerate(data[:-1]):
        ascii = ord(ele)
        if ascii >= 48 and ascii <= 57:
            actual_num = int(ele)
        else:
            if ascii in prediction_to_num:
                actual_num = prediction_to_num[ascii]
            else:
                print(f'Enter the number at index {idx}:', end=' ')
                cv2.imshow('cell', cell)
                cv2.waitKey(1000)
                cv2.destroyWindow('cell')
                cv2.waitKey(1)
                actual_num = int(input())
                prediction_to_num[ascii] = actual_num

        predicted_num = predicted_num * 10 + actual_num

    return predicted_num, prediction_to_num