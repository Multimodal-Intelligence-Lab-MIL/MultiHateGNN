import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    # input: batch*length*inChl; output: batch*length*outChl
    def __init__(self, lstmD, inputEmbSz, nodeFtrD):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(inputEmbSz, lstmD)
        self.fc = nn.Linear(lstmD, nodeFtrD)



    def forward(self, x):
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = x.transpose(0, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x

class Text_Model(nn.Module):
    def __init__(self, input_size, hdSz, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, hdSz[0]),
            nn.ReLU(),
            nn.Linear(hdSz[0], hdSz[1]),
            nn.ReLU(),
            nn.Linear(hdSz[1], output_size),
            nn.ReLU()
        )
    def forward(self, xb):
        return self.network(xb)

class wtMLP(nn.Module):
    def __init__(self, input_size, hdSz, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, hdSz[0]),
            nn.ReLU(),
            nn.Linear(hdSz[0], hdSz[1]),
            nn.ReLU(),
            nn.Linear(hdSz[1], output_size),
        )
        self.softmax = nn.Softmax(dim=0)
    def forward(self, xb, numMod, numNodesInMod, istnsIds):
        nodeSrs = self.network(xb) #nodeSrs: [batch, number of nodes, 1]
        if numMod == 3:
            nodeSrsM1, nodeSrsM2, nodeSrsM3 = nodeSrs[0,:numNodesInMod,0], nodeSrs[0, numNodesInMod: 2 * numNodesInMod, 0], nodeSrs[0, 2 * numNodesInMod:, 0]
            nodeWtsM1, nodeWtsM2, nodeWtsM3 = self.softmax(nodeSrsM1), self.softmax(nodeSrsM2), self.softmax(nodeSrsM3)
            nodeWts = nodeWtsM1 + nodeWtsM2 + nodeWtsM3
            istnsWts = torch.cat([(1/3) * nodeWts[istnsIds[i] : istnsIds[i + 1]].sum().reshape(1) for i in range(istnsIds.shape[0] - 1)])
        elif numMod == 2:
            nodeSrsM1, nodeSrsM2 = nodeSrs[0,:numNodesInMod,0], nodeSrs[0, numNodesInMod: 2 * numNodesInMod, 0]
            nodeWtsM1, nodeWtsM2 = self.softmax(nodeSrsM1), self.softmax(nodeSrsM2)
            nodeWts = nodeWtsM1 + nodeWtsM2
            istnsWts = torch.cat([(1/2) * nodeWts[istnsIds[i] : istnsIds[i + 1]].sum().reshape(1) for i in range(istnsIds.shape[0] - 1)])
        else:
            nodeWts= self.softmax(nodeSrs[0,:,0])
            istnsWts = torch.cat([nodeWts[istnsIds[i] : istnsIds[i + 1]].sum().reshape(1) for i in range(istnsIds.shape[0] - 1)])
        return istnsWts

class classifier(nn.Module):
    def __init__(self, input_size, hdSz, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, hdSz[0]),
            nn.ReLU(),
            nn.Linear(hdSz[0], hdSz[1]),
            nn.ReLU(),
            nn.Linear(hdSz[1], output_size),
        )
    def forward(self, isWts, isFtrs):
        ftrCls = isWts.reshape(1,-1) @ isFtrs
        x = self.network(ftrCls)
        return x

class attMLP(nn.Module):
    def __init__(self, input_size, hdSz, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, hdSz[0]),
            nn.ReLU(),
            nn.Linear(hdSz[0], hdSz[1]),
            nn.ReLU(),
            nn.Linear(hdSz[1], output_size),
        )
    def forward(self, xb):
        return self.network(xb)

class linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network= nn.Linear(input_size, output_size)


    def forward(self, xb):
        return self.network(xb)

