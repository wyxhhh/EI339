import sys
import time

# all the possible action set
actions = {}
actions[(1, 2)] = 0b100000000000000000
actions[(1, 3)] = 0b010000000000000000
actions[(2, 3)] = 0b001000000000000000
actions[(2, 4)] = 0b000100000000000000
actions[(2, 5)] = 0b000010000000000000
actions[(3, 5)] = 0b000001000000000000
actions[(3, 6)] = 0b000000100000000000
actions[(4, 5)] = 0b000000010000000000
actions[(4, 7)] = 0b000000001000000000
actions[(4, 8)] = 0b000000000100000000
actions[(5, 6)] = 0b000000000010000000
actions[(5, 8)] = 0b000000000001000000
actions[(5, 9)] = 0b000000000000100000
actions[(6, 9)] = 0b000000000000010000
actions[(6, 10)] = 0b000000000000001000
actions[(7, 8)] = 0b000000000000000100
actions[(8, 9)] = 0b000000000000000010
actions[(9, 10)] = 0b000000000000000011



class TriWar:
    global actions
    def __init__(self, init_action):
        self.Triangles = (0b111000000000000000, 0b000110010000000000, 0b001011000000000000, 0b000001100010000000, 0b000000001100000100, 0b000000010101000000, 0b000000000001100010, 0b000000000010110000, 0b000000000000011001)
        # all the triangles, the value of which is the logic 'or' of three action-value that are likely to form a triangle
        self.taken = init_action
        # the pre-occur actions form input
        self.Arwd = 0
        self.Brwd = 0
        self.turn = True
        # True incidates A's turn
        self.init_state = 0b000000000000000000
        self.end = 0b111111111111111111

    def TriangleIncrem(self, old, act):
        '''
        the function compares the number of triangles after a new action is taken with the previous one.
        the number is calculated via bit-wise and operation.
        '''
        now= old | act
        cnt = 0
        for tri in self.Triangles:
            if (old&tri != tri) and (now & tri == tri):
                cnt += 1
        return now, cnt

    def minSearch(self, state, beta, Arwd, Brwd):
        v = 1
        if Arwd >= 5:  # if a gets more than 5 triangles, then the game is over (a doomed to win) because total number=9
            return 1
        elif Brwd >= 5:  # vice versa for b
            return -1
        if state == self.end:  # if game is over
            return 1 if Arwd > Brwd else -1
        available = self.end - state
        # check all the possible actions
        while available: 
            act = available & ( - available)
            NewCnt = Brwd
            NewState, increm = self.TriangleIncrem(state, act)
            NewCnt += increm
            tmp = self.minSearch(NewState, beta, Arwd, NewCnt) if NewCnt > Brwd else self.maxSearch(NewState, v, Arwd, Brwd)
            v = min(tmp, v)
            if tmp <= beta:
                return v
            available -= act
        return v

    def maxSearch(self, state, alpha, Arwd, Brwd):
        v = -1
        if Arwd >= 5:      
            return 1
        elif Brwd >= 5:   
            return -1
        if state == self.end: 
            return 1 if Arwd > Brwd else -1
        available = self.end - state
        
        while available:
            act = available & (-available)
            NewCnt = Arwd
            NewState, increm = self.TriangleIncrem(state, act)
            NewCnt += increm
            tmp = self.maxSearch(NewState, alpha, NewCnt, Brwd) if NewCnt > Arwd else self.minSearch(NewState, v, Arwd, Brwd)
            v = max(tmp, v)
            if tmp >= alpha:
                return v
            available -= act
        return v


    def search(self):
        # initialize A's reward , B's reward and whose turn to take next action
        for i in range(len(self.taken)):
            act = self.taken[i] 
            self.init_state, increm = self.TriangleIncrem(self.init_state, actions[act])
            if self.turn:
                self.Arwd += increm
            else:
                self.Brwd += increm
            if not increm:
                self.turn = not self.turn
        if self.turn: # if A's turn
            return self.maxSearch(self.init_state, 1, self.Arwd, self.Brwd)
        else:
            return self.minSearch(self.init_state, -1, self.Arwd, self.Brwd)


if __name__ == "__main__":
    # --------------
    # test data
    # [(2, 4), (4, 5), (5, 9), (3, 6), (2, 5), (3, 5)]  B
    # [(2, 4), (4, 5), (5, 9), (3, 6), (2, 5), (3, 5), (7, 8)]  A
    # [(1, 2), (2, 3), (1, 3), (2, 4), (2, 5), (4, 5)]  A
    # [(1, 2), (2, 5), (3, 6), (5, 8), (4, 7), (6, 10), (2, 4), (4, 5), (4, 8), (7, 8)] 
    # --------------
    args = sys.argv[1:]
    explored = []
    tmp = []
    for i in range(len(args)):
        tmp.append(int(args[i]))
        if i % 2 != 0:
            explored.append(tuple(tmp))
            tmp = []
    search = TriWar(explored)
    result = search.search()
    if result > 0:
        print("A")
    else:
        print("B")
