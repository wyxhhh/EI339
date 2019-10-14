import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import heapq
from queue import PriorityQueue, Queue


class ComparAble:
    """
    This class defines the data structure of priority queue.
    We use a minmum priority queue here.
    pos is the position of a cell in the graph.
    """
    def __init__(self, obj, pos):
        self.priority = obj
        self.pos = pos

    def __lt__(self, other):
        if self.priority <= other.priority:
            return True
        else:
            return False


class MazeProblem:
    def __init__(self, maze_file = '', start=(0, 0), end=(26, 39)):
        self.start = start
        self.end = end
        self.map, self.hrc = self.loadMap(maze_file)
        # self.hrc stores the heuristic estimate h(s) of the graph
        self.exp, self.dict = self.Astar()
        # exp is the set of explored cells set, dict stores the previous cell of one explored cell.
        self.path = self.FindPath()
        # path is the final optimal path from the start cell to the end.

    
    def loadMap(self, file):
        # Load map txt as matrix.
        # 0: path, 1: obstacle, 2: start point, 3: end point
        f = open(file)
        lines = f.readlines()
        numOfLines = len(lines)
        returnMap = np.zeros((numOfLines, 40))
        hrcmap = np.zeros(returnMap.shape)
        A_row = 0
        for line in lines:
            list = line.strip().split(' ')
            returnMap[A_row:] = list[0:40]
            A_row += 1
        width, height = hrcmap.shape
        for x in range(width):
            for y in range(height):
                hrcmap[x, y] = abs(x - width + 1) + abs(y - height + 1)
        # h(s) in this graph is the Manhattan distance to the end point.
        
        returnMap += 1
        returnMap[self.start[0], self.start[1]] = 1
        returnMap[self.end[0], self.end[1]] = 1
        # To simplify the original problem, we define cost of every edge is 1.
        return returnMap, hrcmap

    def is_valid(self, pos):
        x, y = pos
        return not (x < 0 or x >= self.map.shape[0] or y < 0 or y >= self.map.shape[1] or self.map[x, y] == 2)

    def Astar(self):
        exp = []
        ftr = PriorityQueue()
        (x, y) = self.start
        path = {}
        bias = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        # ftr is the frontier priority queue, bias is the possible successor cells

        ftr.put(ComparAble(self.map[x, y], (x, y)))
        # delete the min cost cell in the priority queue
        cnt = 0
        path[(x, y)] = None
        while ftr.qsize() != 0:
            cnt += 1
            s = ftr.get()
            prio = s.priority
            # prio is the current minimum cost.

            exp.append(s.pos)
            if s.pos == self.end:
                print(cnt)
                break
            # if the end point is in the explored set, break loop
            x, y = s.pos

            for (x_bias, y_bias) in bias:
                curr_x = x + x_bias
                curr_y = y + y_bias
                # (curr_x, curr_y) is the coordinate of successing cells.

                if self.is_valid((curr_x, curr_y)):
                    tpl = (curr_x, curr_y)
                    if tpl in exp:
                        continue
                    else:
                        lcprio = prio + self.map[x, y] + self.hrc[curr_x, curr_y] - self.hrc[x, y]
                        # update the cost of every successing cells using heuristic estimate.
                        pos = (curr_x, curr_y)
                        ftr.put(ComparAble(lcprio, pos))
                        # push all the successing cells into the priority queue.
                        path[pos] = (x, y)

        self.map -= 1
        self.map[self.start[0], self.start[1]] = 2
        self.map[self.end[0], self.end[1]] = 3
        # modify the value and state of the input matrix to its origin.
        return exp, path

    def FindPath(self):
        '''
        This function finds the optimal path from origin to the end.
        Implemented by recursively find the previous cell of the current one.
        '''
        origin = self.start
        print(origin)
        end = self.end
        path = []

        while origin != end:
            end = self.dict[end]
            path.append(end)
        print(len(path))
        return path

    def drawMap(self):
        # Visulize the maze map.
        # Draw obstacles(1) as red rectangles. Draw path(0) as white rectangles. Draw starting point(2) and ending point(3) as circles.
        rowNum = len(self.map)
        # print(rowNum)
        colNum = len(self.map[0])
        # print(colNum)
        ax = plt.subplot()
        param = 1
        for col in range(colNum):
            for row in range(rowNum):
                if self.map[row, col] == 3:
                    circle = mpathes.Circle([param * col + param/2.0, param * row + param/2.0], param/2.0, color='g')
                    ax.add_patch(circle)
                elif self.map[row,col] == 2:
                    circle = mpathes.Circle([param * col + param/2.0, param * row + param/2.0], param/2.0, color='y')
                    ax.add_patch(circle)
                elif self.map[row, col] == 1:
                    rect = mpathes.Rectangle([param * col, param * row], param, param, color='r')
                    ax.add_patch(rect)
                else:
                    rect = mpathes.Rectangle([param * col, param * row], param, param, color='w')
                    ax.add_patch(rect)
        for item in self.path:
            print(item)
            rect = mpathes.Rectangle([param * item[1], param * item[0]], param, param, color='g')
            ax.add_patch(rect)
        # Improve visualization
        plt.xlim((0,colNum))
        plt.ylim((0,rowNum))
        my_x_ticks = np.arange(0,colNum+1, 1)
        my_y_ticks = np.arange(0,rowNum+1, 1)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        axx = plt.gca()
        axx.xaxis.set_ticks_position('top')
        axx.invert_yaxis()
        plt.grid()
        # Save maze image.
        plt.savefig('maze1.jpg')

if __name__ == "__main__":
    Solution = MazeProblem(maze_file = 'maze.txt', start=(0, 0), end=(26, 39))
    Solution.drawMap()




