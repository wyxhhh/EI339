import numpy as np
import sys
from functools import lru_cache

# all the possible actions for agent A(white) and agent B(black)
actions = {}
actions['w'] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
actions['b'] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]

def move(pos1, pos2):
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])

class Chess: 
    def __init__(self, r, c, name):
        self.Pos = (r, c)
        self.name = name


class ChessBoard(Chess):
    def __init__(self, n, r1, c1, r2, c2):
        self.size = n
        self.white = Chess(r1, c1, 'w')
        self.Black = Chess(r2, c2, 'b')

    def is_valid(self, pos): # whether the position is in the chess board
        return not (pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size)

    def IsEnd(self, A, B): # whether A or B wins
        return A[0] == B[0] and A[1] == B[1]

    @lru_cache(None)
    def maxvalue(self, Maxstep, depth, w_pos, b_pos):
        if self.IsEnd(w_pos, b_pos):
            return depth - 100
        if depth > 4 * Maxstep: # limit the search depth
            return -100
        v = -100
        actlst = []
        for act in actions['w']: # transverse all the valid actions
            if self.is_valid(move(w_pos, act)):
                actlst.append(act)
        for act in actlst:
            a = self.minvalue(Maxstep, depth+1, move(w_pos, act), b_pos)
            if a > v:
                v = a
        return v

    @lru_cache(None)
    def minvalue(self, Maxstep, depth, w_pos, b_pos):
        if self.IsEnd(w_pos, b_pos):
            return depth + 100
        if depth > 4 * Maxstep:
            return 100
        v = 100
        actlst = []
        for act in actions['b']:
            if self.is_valid(move(b_pos, act)):
                actlst.append(act)
        for act in actlst:
            a = self.maxvalue(Maxstep, depth+1, w_pos, move(b_pos, act))
            if v > a:
                v = a
        return v

    
    def MinMaxSearch(self, MaxDepth, depth):
        de = self.maxvalue(MaxDepth, depth, self.white.Pos, self.Black.Pos)
        return de


if __name__ == "__main__":
    n, r1, c1, r2, c2 = [int(item) for item in sys.argv[1:]]
    if (r1 == r2 and abs(c1 - c2) == 1) or (c1 == c2 and abs(r1 - r2)==1):
        print("WHITE 1")
    else:
        game = ChessBoard(int(n), int(r1-1), int(c1-1), int(r2-1), int(c2-1))
        num = game.MinMaxSearch(n, 0)
        print("BLACK {}".format(100+num))

