import numpy as np


def _get_nodes(puzzle: np.ndarray):
    """
    A function to convert a puzzle to an array of nodes. Here, a node contains
    the number from the current puzzle. The order of the nodes is purely
    sequential. Every number observed sequentially gets appended to the list
    of nodes before being returned. If a hint cell contains two numbers,
    then two nodes would be added to the nodes list.
    :param puzzle: the instance of the kakuro puzzle.
    :type puzzle: `numpy.ndarray`
    TODO: Update the return values.
    :return: the nodes in the puzzle
    :returns: `numpy.ndarray`
    """
    nodes = np.array([])

    # the below array basically contains details whether the current node
    # is part of a double hint cell. If yes, corresponding index of the node
    # in the cell would be 1. Else 0.
    double_hint_cell = np.array([])

    for r_idx, row in enumerate(puzzle):
        for c_idx, ele in enumerate(row):
            if ele.isnum():
                nodes = np.append(nodes, int(ele))
                double_hint_cell = np.append(double_hint_cell, 0)


def generate_adjacency_list_for_puzzle(puzzle: np.ndarray):
    """
    A function to generate the adjacency list for a puzzle given its array
    representation.
    :param puzzle: an instance of the kakuro puzzle.
    :type puzzle: `numpy.ndarray`
    :return: the adjacency list
    :returns: `numpy.ndarray`
    """
    nodes, double_hint_cell = np.array([])

    # the below array basically contains details whether the current node
    # is part of a double hint cell. If yes, corresponding index of the node
    # in the cell would be 1. Else 0.
    double_hint_cell = np.array([])
    for r_idx, row in enumerate(puzzle):
        for c_idx, ele in enumerate(row):
            # keep this on hold
            if '\\' in ele:
                left_hint, right_hint = ele.split('\\')
                if not left_hint.isnum() and not right_hint.isnum():
                    continue
                if left_hint.isnum():
                    left_hint = int(left_hint)