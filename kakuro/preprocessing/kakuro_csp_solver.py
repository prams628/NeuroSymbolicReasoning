import sys
import random
# from Tkinter import Tk, Canvas, Frame, Button, BOTH, TOP, RIGHT
from datetime import datetime
from pulp import *

random.seed(datetime.now())

MARGIN = 20
SIDE = 50
WIDTH = HEIGHT = MARGIN * 2 + SIDE * 9

class KakuroError(Exception):
    """
    An application specific error.
    """
    pass

class KakuroUI:
    """
    The Tkinter UI: draw the board and accept input
    """
    def __init__(self, game):
        self.game = game
        self.row, self.col = -1, -1


    def solve(self):
        self.game.data_filled = []
        options = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        # Remember to zero down the indices
        vals = options
        rows = options
        cols = options
        prob = LpProblem("Kakuro_Problem", LpMinimize)
        choices = LpVariable.dicts("Choice", (vals, rows, cols), 0, 1, LpInteger)
        # The complete set of boolean choices
        prob += 0, "Arbitrary Objective Function"
        # Force singular values. Even for extraneous ones
        for r in rows:
            for c in cols:
                prob += lpSum([choices[v][r][c] for v in vals]) == 1, ""

        # Force uniqueness in each 'zone' and sum constraint for that zone
        # Row zones
        for r in rows:
            zonecolsholder = []
            activated = False
            zonecolssumholder = 0
            for c in cols:
                if not activated and [int(r)-1,int(c)-1] in self.game.data_fills:
                    activated = True
                    zonecolsholder = zonecolsholder +[c]
                    zonecolssumholder = [elem[0] for elem in self.game.data_totals if elem[1]=='h' and elem[2] == int(r)-1 and elem[3] == int(c)-2][0]
                    if c == "9":
                        # Uniqueness in that zone of columns
                        for v in vals:
                            prob += lpSum([choices[v][r][co] for co in zonecolsholder]) <= 1, ""
                        # Sum constraint for that zone of columns
                        prob += lpSum([int(v) * choices[v][r][co] for v in vals for co in zonecolsholder]) == zonecolssumholder, ""
                elif activated and [int(r)-1,int(c)-1] in self.game.data_fills:
                    zonecolsholder = zonecolsholder + [c]
                    if c == "9":
                        # Uniqueness in that zone of columns
                        for v in vals:
                            prob += lpSum([choices[v][r][co] for co in zonecolsholder]) <= 1, ""
                        # Sum constraint for that zone of columns
                        prob += lpSum([int(v) * choices[v][r][co] for v in vals for co in zonecolsholder]) == zonecolssumholder, ""
                elif activated and [int(r)-1,int(c)-1] not in self.game.data_fills:
                    activated = False
                    # Uniqueness in that zone of columns
                    for v in vals:
                        prob += lpSum([choices[v][r][co] for co in zonecolsholder]) <= 1, ""
                    # Sum constraint for that zone of columns
                    prob += lpSum([int(v)*choices[v][r][co] for v in vals for co in zonecolsholder]) == zonecolssumholder, ""
                    zonecolssumholder = 0
                    zonecolsholder = []

        # Col zones
        for c in cols:
            zonerowsholder = []
            activated = False
            zonerowssumholder = 0
            for r in rows:
                if not activated and [int(r)-1,int(c)-1] in self.game.data_fills:
                    activated = True
                    zonerowsholder = zonerowsholder +[r]
                    zonerowssumholder = [elem[0] for elem in self.game.data_totals if elem[1]=='v' and elem[2] == int(r)-2 and elem[3] == int(c)-1][0]
                    if r == "9":
                        # Uniqueness in that zone of rows
                        for v in vals:
                            prob += lpSum([choices[v][ro][c] for ro in zonerowsholder]) <= 1, ""
                        # Sum constraint for that zone of rows
                        prob += lpSum([int(v) * choices[v][ro][c] for v in vals for ro in zonerowsholder]) == zonerowssumholder, ""
                elif activated and [int(r)-1,int(c)-1] in self.game.data_fills:
                    zonerowsholder = zonerowsholder + [r]
                    if r == "9":
                        # Uniqueness in that zone of rows
                        for v in vals:
                            prob += lpSum([choices[v][ro][c] for ro in zonerowsholder]) <= 1, ""
                        # Sum constraint for that zone of rows
                        prob += lpSum([int(v) * choices[v][ro][c] for v in vals for ro in zonerowsholder]) == zonerowssumholder, ""
                elif activated and [int(r)-1,int(c)-1] not in self.game.data_fills:
                    activated = False
                    # Uniqueness in that zone of rows
                    for v in vals:
                        prob += lpSum([choices[v][ro][c] for ro in zonerowsholder]) <= 1, ""
                    # Sum constraint for that zone of rows
                    prob += lpSum([int(v)*choices[v][ro][c] for v in vals for ro in zonerowsholder]) == zonerowssumholder, ""
                    zonerowssumholder = 0
                    zonerowsholder = []

        # Force all extraneous values to 1 (arbitrary) | Possibly many times
        for ite in self.game.data_totals:
            prob += choices["1"][str(ite[2]+1)][str(ite[3]+1)] == 1, ""

        # Suppress calculation messages
        prob.solve(PULP_CBC_CMD(timeLimit=1000, msg=0, gapRel=0))
        # Solution: The commented print statements are for debugging aid.
        for v in prob.variables():
            # print v.name, "=", v.varValue
            if v.varValue == 1 and [int(v.name[9])-1, int(v.name[11])-1] in self.game.data_fills:
                # print v.name, ":::", v.varValue, [int(v.name[9])-1, int(v.name[11])-1, int(v.name[7])]
                self.game.data_filled = self.game.data_filled + [[int(v.name[9])-1, int(v.name[11])-1, int(v.name[7])]]


class KakuroCustomGame(object):
    """
    A Kakuro game. Stores gamestate and completes the puzzle as needs be
    """
    def __init__(self, puzzle):
        self.played_so_far = []
        self.data_filled = []
        self.data_fills = []
        self.data_totals = []
        try:
            for i in range(9):
                text = puzzle.split('\n')[i]
                proced = [ele.split('\\') for ele in text.split(',')]
                if len(proced)!=9:
                    raise KakuroError('Nine cells a line or else format not followed!\n')
                for j in range(9):
                    if len(proced[j]) == 1 and proced[j][0] == ' ':
                        self.data_fills = self.data_fills + [[i,j]]
                    elif len(proced[j]) == 2:
                        if proced[j][0]!=' ':
                            self.data_totals = self.data_totals + [[int(proced[j][0]),'v',i,j]]
                        if proced[j][1]!=' ':
                            self.data_totals = self.data_totals + [[int(proced[j][1]),'h',i,j]]
        except(ValueError):
            raise KakuroError('Format not followed! Integers only otherwise something else')
        self.gameId = 0
        self.game_over = False

    def check_win(self):
        if(len(self.data_filled) == len(self.data_fills)):
            for item in self.data_filled:
                if [item[0], item[1]-1] not in self.data_fills:
                    sumexp = -1
                    for elem in self.data_totals:
                        if elem[2] == item[0] and elem[3] == item[1]-1 and elem[1] == 'h':
                            sumexp = elem[0]
                    offset = 0
                    sumact = []
                    while [item[0], item[1]+offset] in self.data_fills:
                        sumact = sumact + [e[2] for e in self.data_filled if e[0] == item[0] and e[1] == item[1]+offset]
                        offset = offset + 1
                    if len(sumact) != len(set(sumact)):
                        return False
                    if sumexp != -1 and sumexp != sum(sumact):
                        return False
            for item in self.data_filled:
                if [item[0]-1, item[1]] not in self.data_fills:
                    sumexp = -1
                    for elem in self.data_totals:
                        if elem[2] == item[0]-1 and elem[3] == item[1] and elem[1] == 'v':
                            sumexp = elem[0]
                    offset = 0
                    sumact = []
                    while [item[0]+offset, item[1]] in self.data_fills:
                        sumact = sumact + [e[2] for e in self.data_filled if e[0] == item[0]+offset and e[1] == item[1]]
                        offset = offset + 1
                    if len(sumact) != len(set(sumact)):
                        return False
                    if sumexp != -1 and sumexp != sum(sumact):
                        return False
            return True
        else:
            return False
