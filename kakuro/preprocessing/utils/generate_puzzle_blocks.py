import numpy as np


def _pad_sequence(sequence: np.ndarray, length_after_padding: int, pad_with: int) -> np.ndarray:
    """
    Pads `sequence` with `pad_with` so that the length is equal to
    `length_after_padding`.
    """
    pad_array = np.array([pad_with for _ in range(length_after_padding - len(sequence))])
    return np.append(
        sequence,
        pad_array
    )


def _get_curr_block(puzzle: np.ndarray, curr_block: np.ndarray, c_idx: int, r_idx: int, hint_type: str):
    """
    TODO: Complete the documentation
    """
    if hint_type == 'left':
        while puzzle[r_idx][c_idx].isnumeric():
            curr_block = np.append(curr_block, int(puzzle[r_idx][c_idx]))
            r_idx += 1
    elif hint_type == 'right':
        while puzzle[r_idx][c_idx].isnumeric():
            curr_block = np.append(curr_block, int(puzzle[r_idx][c_idx]))
            c_idx += 1
    return curr_block


def generate_puzzle_blocks(puzzle: np.ndarray) -> np.ndarray:
    """
    Generates a sequence of blocks from the puzzle which form a sequence.
    TODO: Complete the documentation
    """
    block_length = len(puzzle)
    blocks = np.empty((0, block_length))
    for r_idx, row in enumerate(puzzle):
        for c_idx, ele in enumerate(row):
            if '\\' in ele:
                left_hint, right_hint = ele.split('\\')
                if left_hint.isnumeric():
                    curr_block = np.array([int(left_hint)])
                    curr_block = _get_curr_block(puzzle, curr_block, c_idx, r_idx + 1, 'left')
                    curr_block = _pad_sequence(
                        sequence=curr_block,
                        length_after_padding=block_length,
                        pad_with=-1
                    )
                    blocks = np.vstack((blocks, curr_block))
                if right_hint.isnumeric():
                    curr_block = np.array([int(right_hint)])
                    curr_block = _get_curr_block(puzzle, curr_block, c_idx + 1, r_idx, 'right')
                    curr_block = _pad_sequence(
                        sequence=curr_block,
                        length_after_padding=block_length,
                        pad_with=-1
                    )
                    blocks = np.vstack((blocks, curr_block))
    return blocks if _validate_solution(blocks) else None


def _validate_solution(solution: np.ndarray):
    """
    TODO: Update documentation
    """
    for row in solution:
        row_sum = 0
        for idx in range(1, len(row)):
            if row[idx] == -1:
                break
            row_sum += row[idx]
        if row_sum != row[0]:
            return False
    return True