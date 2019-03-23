import numpy as np
import sys

from termcolor import colored, cprint

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batchnorm = nn.BatchNorm2d(3, affine=False)
        self.pad2 = nn.ConstantPad2d(2, 0)
        self.pad1 = nn.ConstantPad2d(1, 0)

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 16, 3)
        self.conv5 = nn.Conv2d(16, 1, 5)

    def forward(self, x):
        x = self.batchnorm(x.float())
        x = self.pad1(F.relu(self.conv1(x)))
        x = self.pad1(F.relu(self.conv2(x)))
        x = self.pad1(F.relu(self.conv3(x)))
        x = self.pad1(F.relu(self.conv4(x)))
        x = self.pad2(F.relu(self.conv5(x)))

        x = x.view(-1, 225)
 
        return x


def toTripleBoard(board, side):
    curBoard = np.zeros(shape=(3, 15, 15), dtype=np.float)
    blackBoard = np.zeros(shape=(15, 15), dtype=np.float)
    whiteBoard = np.zeros(shape=(15, 15), dtype=np.float)

    if side == 1:
        blackTurnBoard = np.ones(shape=(15, 15), dtype=np.float)
    else:
        blackTurnBoard = -np.ones(shape=(15, 15), dtype=np.float)
    curBoard[0,:] = blackTurnBoard
    for i in range(15):
        for j in range(15):
            if board[i, j] == 1:
	            blackBoard[i, j] = 1
            if board[i, j] == -1:
                whiteBoard[i, j] = -1
    curBoard[1,:] = blackBoard
    curBoard[2,:] = whiteBoard
    return curBoard


def isGg(board, color):
    if color == 0:
        size = 4
    else:
        size = -4
    for i in range(15):
        for j in range(10):
            if board[i, j] + board[i, j + 1] + board[i, j + 2] + board[i, j + 3] + board[i, j + 4] == size:
                for k in range(5):
                    if board[i, j + k] == 0:
                        return i, j + k

    for i in range(10):
        for j in range(15):
            if board[i, j] + board[i + 1, j] + board[i + 2, j] + board[i + 3, j] + board[i + 4, j] == size:
                for k in range(5):
                    if board[i + k, j] == 0:
                        return i + k, j
    
    for i in range(10):
        for j in range(10):
            if board[i, j] + board[i + 1, j + 1] + board[i + 2, j + 2] + board[i + 3, j + 3] + board[i + 4, j + 4] == size:
                for k in range(5):
                    if board[i + k, j + k] == 0:
                        return i + k, j + k
    for i in range(5, 15):
        for j in range(10):
            if board[i, j] + board[i - 1, j + 1] + board[i - 2, j + 2] + board[i - 3, j + 3] + board[i - 4, j + 4] == size:
                for k in range(5):
                    if board[i - k, j + k] == 0:
                        return i - k, j + k
    return (-1, -1)



def toTurn(turn):
    letter = ord(turn[0]) - ord('a')
    num = int(turn[1:]) - 1
    return letter, num

def to_move(pos):
    return idx2chr[pos[0]] + str(pos[1] + 1)

def to_pos(move):
    return chr2idx[move[0]], int(move[1:]) - 1

if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.backends.cudnn.benchmark = True
	idx2chr = 'abcdefghjklmnop'
	chr2idx = {letter: pos for pos, letter in enumerate(idx2chr)}
	net = Net()
	net = net.to(device)
	net.load_state_dict(torch.load("finalNN.txt"))

	while True:
		gameStr = sys.stdin.readline()#input()#
		if not gameStr:
			break
		gameStr = gameStr.strip().split()
		curBoard = np.zeros(shape=(15, 15), dtype=np.float)

		for i in range(len(gameStr)):
		    x, y = to_pos(gameStr[i])
		    if i % 2 == 0:
		        curBoard[x, y] = 1
		    else:
		        curBoard[x, y] = -1

		color = len(gameStr) % 2


		board = toTripleBoard(curBoard, color)
		with torch.no_grad():
		    outputs = net(torch.unsqueeze(torch.from_numpy(board), 0))
		    _, netTurn = torch.max(outputs, 1)
		    netTurn = int(netTurn)
		    turnX, turnY = netTurn // 15, netTurn % 15
		    while curBoard[turnX, turnY] != 0:
		        outputs[netTurn] = 0
		        _, netTurn = torch.max(outputs, 1)
		        netTurn = int(netTurn)
		        turnX, turnY = netTurn // 15, netTurn % 15

		ggTest = isGg(curBoard, color)
		if ggTest != (-1, -1):
		    turnX, turnY = ggTest

		myTurn = to_move((turnX, turnY))
		sys.stdout.write(myTurn + '\n')
		sys.stdout.flush()




















