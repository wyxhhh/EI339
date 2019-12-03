import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.pyplot import MultipleLocator


class Environment(object):
    def __init__(self, rows=5, cols=8, barrier_num=3, reward_num=5):
        self.rows = rows
        self.cols = cols
        self.barrier_num = barrier_num
        self.reward_num = reward_num
        self.create_env_default()

    def create_env_default(self):
        # the final matrix of environment in our problem
        # start node = 1, terminal node = 2, cliff = -1, barrier = -2, reward = 3
        # [[1. - 1. - 1. - 1. - 1. - 1. - 1.  2.]
        # [0.  0.  0.  0.  0.  0. - 2.  0.]
        # [0.  0.  0.  3. - 2.  0.  0.  0.]
        # [3.  0.  0.  0. - 2.  0.  0.  3.]
        # [0.  0.  3.  0.  0.  0.  0.  0.]]

        self.env = np.zeros([self.rows, self.cols])
        # start node = 1, terminal node = 2, cliff = -1
        self.env[0][0] = 1
        self.env[0][self.cols-1] = 2
        self.env[0][1:self.cols-1] = -1

        # set barrier pos
        barrier_pos = [[3, 4], [2, 4], [1, 6]]
        # set barrier = -2
        for pos in barrier_pos:
            self.env[pos[0]][pos[1]] = -2

        # set reward pos
        reward_pos = [[3, 0], [2, 3], [4, 2], [3, 7]]
        # set reward = 3
        for pos in reward_pos:
            self.env[pos[0]][pos[1]] = 3

    def create_env(self):
        self.env = np.zeros([self.rows, self.cols])
        # start node = 1, terminal node = 2, cliff = -1
        self.env[0][0] = 1
        self.env[0][self.cols-1] = 2
        self.env[0][1:self.cols-1] = -1

        # randomly set barrier pos
        barrier_pos = []
        while(len(barrier_pos) < self.barrier_num):
            i = random.randint(1, self.rows-1)
            j = random.randint(0, self.cols-1)
            if [i, j] not in barrier_pos and [i, j] not in [[1, 0], [1, self.cols-1]]:
                barrier_pos.append([i, j])

        # set barrier = -2
        for pos in barrier_pos:
            self.env[pos[0]][pos[1]] = -2

        # randomly set reward pos
        reward_pos = []
        while (len(reward_pos) < self.reward_num):
            i = random.randint(1, self.rows - 1)
            j = random.randint(0, self.cols - 1)
            if [i, j] not in reward_pos and [i, j] not in barrier_pos:
                reward_pos.append([i, j])

        # set reward = 3
        for pos in reward_pos:
            self.env[pos[0]][pos[1]] = 3

    def show_env(self, sarsa):
        # fig = plt.figure()
        ax = plt.subplot()
        plt.xlim((0, self.cols))
        plt.ylim((0, self.rows))
        # name: start, terminal, cliff, barrier, reward, others
        # number: 1, 2, -1, -2, 3, 0
        # color: yellow, orange, gray, black, red, white
        color_dict = {-1:"gray", 1:"yellow", 2:"orange", -2:"black", 3:"red", 0:"white"}
        my_x_ticks = np.arange(0, self.cols, 1)
        my_y_ticks = np.arange(0, self.rows, 1)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()

        plt.grid()
        path = sarsa.GetPath()
        for i in range(self.rows):
            for j in range(self.cols):
                color = color_dict[int(self.env[i][j])]
                # if the cell is in the path we derive, I mark them in dodgerblue
                rect = mpathes.Rectangle([j, i], 1, 1, color=color) if (i, j) not in path else mpathes.Rectangle([j, i], 1, 1, color='dodgerblue')
                ax.add_patch(rect)
                # draw the arrows
                direction = np.argmax(sarsa.Q[i, j, :])
                if direction == 0:
                    arrow = mpathes.Arrow(j + 0.5, i + 1, 0, -0.8, width=0.1, color='black')
                elif direction == 1:
                    arrow = mpathes.Arrow(j + 0.5, i, 0, 0.8, width=0.1, color='black')
                elif direction == 2:
                    arrow = mpathes.Arrow(j + 1, i + 0.5, -0.8, 0, width=0.1, color='black')
                elif direction == 3:
                    arrow = mpathes.Arrow(j, i + 0.5, 0.8, 0, width=0.1, color='black')
                ax.add_patch(arrow)
        plt.show()


class Sarsa():
    def __init__(self, env, maxnum):
        self.env = env.env
        self.rows = env.rows
        self.reward = np.zeros((env.rows, env.cols))
        self.Reward()
        self.cols = env.cols
        self.Q = np.zeros((self.rows, self.cols, 4))
        self.upnum = np.zeros(self.Q.shape)
        self.actions = []
        self.ActionLst()
        self.epsilon = 0.1
        self.learning(maxnum)
        self.ZeroDeal()
        self.GetPath()

    def ActionLst(self):
        # Initialize the action list up:(-1,0), down:(1,0). left:(0,-1), right:(0,1)
        self.actions.append((-1, 0))
        self.actions.append((1, 0))
        self.actions.append((0, -1))
        self.actions.append((0, 1))

    def EpsilonDecay(self):
        # decrease epsilon by 0.6
        self.epsilon *= 0.6

    def Reward(self):
        # Define the reward in the grid
        self.reward[0, 7] = 10
        self.reward[0, 1 : 7] = -100
        self.reward[3, 0], self.reward[4, 2], self.reward[2, 3], self.reward[3, 7] = -1, -1, -1, -1

    def barrier(self, pos):
        # Judge whether a cell is a barrier or out of the boundary of the grid
        x, y = pos
        return (x < 0 or x >= self.rows or y < 0 or y >= self.cols or self.env[x, y] == -2)

    def NextState(self, x, y, action):
        # Get the next state s' by taking action a from s
        x_bias, y_bias = action
        Sprime = (x + x_bias, y + y_bias)
        act = 4
        if not self.barrier(Sprime):
            act = self.action2num(action)
        return Sprime, act

    def action2num(self, action):
        # transfer the action to number, 0 for up, 1 for down, 2 for left and 3 for right 
        if action == (-1, 0):
                act = 0
        elif action == (1, 0):
            act = 1
        elif action == (0, -1):
            act = 2
        elif action == (0, 1):
            act = 3
        return act

    def num2action(self, num):
        if num == 0:
            return (-1, 0)
        elif num == 1:
            return (1, 0)
        elif num == 2:
            return (0, -1)
        else:
            return (0, 1)

    def EpsilonGreedy(self, pos):
        # p=epsilon choose a random action and p = 1-epsilon take the action with max Q value
        x, y = pos
        epsilon = random.randint(0, 99)
        if epsilon <= 100 * self.epsilon:
            act = random.choice(self.actions)
            return act
        else:
            act = np.argmax(self.Q[x, y, :])
            if act == 0:
                return (-1, 0)
            elif act == 1:
                return (1, 0)
            elif act == 2:
                return (0, -1)
            else:
                return (0, 1)
        
    def Sarsa(self, eta, x, y, act_num, Sprime, act_num_prime, gamma):
        # update the Q value
        return (1 - eta) * self.Q[x, y, act_num] + eta * (self.reward[Sprime] + gamma * self.Q[Sprime[0], Sprime[1], act_num_prime]) 

    # sarsa learning
    def learning(self, max_episode_num, gamma=0.9):
        # gamma: the discount factor
        # max_episode_num: total episode num
        print("sarsa learning")
        epoch = 0
        while epoch < max_episode_num:
            if epoch > 0 and epoch % (max_episode_num // 3) == 0:
                self.EpsilonDecay
            print('\repoch {}/{}'.format(epoch + 1, max_episode_num), end='')
            sys.stdout.flush()
            x, y = 0, 0
            epsilon = random.randint(0, 9)
            if epsilon <= 100 * self.epsilon:
                act = random.choice([(0, 1), (1, 0)])
            else:
                act = (0, 1) if self.Q[x, y, 1] > self.Q[x, y, 3] else (1, 0)
            Sprime, act_num = self.NextState(x, y, act)
            if act_num == 4: #barrier or out of boundary
                continue
            while self.env[x, y] != 2:
                upnum = self.upnum[x, y, act_num]
                eta = 1/(1 + upnum)
                actprime = self.EpsilonGreedy(Sprime)
                Spp = (actprime[0] + Sprime[0], actprime[1] + Sprime[1])
                '''
                The action in the next state is decided in current state.
                We need to test the validation of both s' and s''
                '''
                while self.barrier(Sprime) or self.barrier(Spp):
                    actprime = self.EpsilonGreedy(Sprime)
                    Spp = (actprime[0] + Sprime[0], actprime[1] + Sprime[1])
                
                act_num_prime = self.action2num(actprime)
                self.Q[x, y, act_num] = self.Sarsa(eta, x, y, act_num, Sprime,act_num_prime, gamma)
                self.upnum[x, y, act_num] += 1
                act = actprime
                (x, y) = Sprime
                Sprime, act_num = self.NextState(x, y, act)
            epoch += 1
    
    def ZeroDeal(self):
        '''
        deal with then Q value list.
        if the max value 
        ''' 
        for x in range(self.rows):
            for y in range(self.cols):
                if np.max(self.Q[x, y, :]) == 0:
                    for i in range(len(self.Q[x, y, :])):
                        if self.Q[x, y, i] == 0:
                            self.Q[x, y, i] = np.min(self.Q[x, y, :])
    
    def GetPath(self):
        pdict = [(0, 0)]
        x, y = 0, 0
        while self.env[x, y] != 2:
            bias = self.num2action(np.argmax(self.Q[x, y, :]))
            x, y = x + bias[0], y + bias[1]
            pdict.append((x, y))
        return pdict


if __name__ == "__main__":
    Env = Environment()
    sarsa = Sarsa(Env, 100000)
    path = sarsa.GetPath()
    print()
    Env.show_env(sarsa)