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
