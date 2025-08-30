import torch
from torch_geometric.nn import GCNConv, GATConv,GATv2Conv, GraphNorm
import torch.nn as nn
import math
import torch.nn.functional as F

# class GCN2L(torch.nn.Module):
#     def __init__(self, inChl, hdChl, outChl, numCls):
#         super(GCN2L, self).__init__()
#         self.conv1 = GCNConv(inChl, hdChl)
#         self.conv2 = GCNConv(hdChl, outChl)
#         self.clf = nn.Linear(outChl, numCls)
#     def forward(self, x, edgeIds):
#         x = self.conv1(x, edgeIds).relu()
#         x = self.conv2(x, edgeIds)
#         x = torch.mean(x, 1)
#         x = self.clf(x)
#         return x


# class GCN2LDp(torch.nn.Module):
#     def __init__(self, inChl, hdChl, outChl, numCls):
#         super(GCN2LDp, self).__init__()
#         self.conv1 = GCNConv(inChl, hdChl)
#         self.conv2 = GCNConv(hdChl, outChl)
#         self.dropout = nn.Dropout()
#         self.clf = nn.Linear(outChl, numCls)
#     def forward(self, x, edgeIds):
#         x = self.conv1(x, edgeIds).relu()
#         x = self.dropout(x)
#         x = self.conv2(x, edgeIds)
#         x = torch.mean(x, 1)
#         x = self.clf(x)
#         return x


class wtGNN(torch.nn.Module):
    def __init__(self, inChl, hdChlList, outChl, numHeads):
        super(wtGNN, self).__init__()
        gatHdChlList = [int(hdChl/numHeads) for hdChl in hdChlList]
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChlList[0], heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChlList[0], out_channels=gatHdChlList[1], heads=numHeads)
        self.conv3 = GATv2Conv(in_channels=hdChlList[1], out_channels=gatHdChlList[2], heads=numHeads)
        self.conv4 = GATv2Conv(in_channels=hdChlList[2], out_channels=gatOutChl, heads=numHeads)
    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds).relu()
        x = self.conv3(x, edgeIds).relu()
        x = self.conv4(x, edgeIds).relu()
        x = torch.unsqueeze(x,dim=0)
        return x

class wtGNNNorm(torch.nn.Module):
    def __init__(self, inChl, hdChlList, outChl, numHeads):
        super(wtGNNNorm, self).__init__()
        gatHdChlList = [int(hdChl/numHeads) for hdChl in hdChlList]
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChlList[0], heads=numHeads)
        self.norm1 = nn.LayerNorm(hdChlList[0])
        self.conv2 = GATv2Conv(in_channels=hdChlList[0], out_channels=gatHdChlList[1], heads=numHeads)
        self.norm2 = nn.LayerNorm(hdChlList[1])
        self.conv3 = GATv2Conv(in_channels=hdChlList[1], out_channels=gatHdChlList[2], heads=numHeads)
        self.norm3 = nn.LayerNorm(hdChlList[2])
        self.conv4 = GATv2Conv(in_channels=hdChlList[2], out_channels=gatOutChl, heads=numHeads)
        self.norm4 = nn.LayerNorm(outChl)
        self.dropout = nn.Dropout()
    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x, edgeIds)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x, edgeIds)
        x = self.norm3(x)
        x = F.relu(x)

        x = self.conv4(x, edgeIds)
        x = self.norm4(x)
        x = F.relu(x)
        x = torch.unsqueeze(x,dim=0)
        return x

class isGNN(torch.nn.Module):
    def __init__(self, inChl, hdChlList, outChl, numHeads):
        super(isGNN, self).__init__()
        gatHdChlList = [int(hdChl/numHeads) for hdChl in hdChlList]
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChlList[0], heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChlList[0], out_channels=gatHdChlList[1], heads=numHeads)
        self.conv3 = GATv2Conv(in_channels=hdChlList[1], out_channels=gatHdChlList[2], heads=numHeads)
        self.conv4 = GATv2Conv(in_channels=hdChlList[2], out_channels=gatOutChl, heads=numHeads)
    def forward(self, x, edgeIds, numSegsVis, numSegsAud, numSegsTx, numNodesInMod, istnsIds, device):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds).relu()
        x = self.conv3(x, edgeIds).relu()
        x = self.conv4(x, edgeIds).relu()
        if numSegsVis > 0 and numSegsAud > 0 and numSegsTx > 0:
             xVis, xAud, xTx = x[:numNodesInMod, :], x[numNodesInMod : 2 * numNodesInMod, :], x[2 * numNodesInMod:, :]
             isFtrsVis = torch.stack([xVis[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
             isFtrsAud = torch.stack([xAud[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
             isFtrsTx = torch.stack([xTx[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
             isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx > 0:
            xAud, xTx = x[:numNodesInMod, :], x[numNodesInMod: 2 * numNodesInMod, :]
            isFtrsAud = torch.stack([xAud[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
            isFtrsTx = torch.stack([xTx[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsVis = torch.zeros_like(isFtrsAud).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx > 0:
            xVis, xTx = x[:numNodesInMod, :], x[numNodesInMod: 2 * numNodesInMod, :]
            isFtrsVis = torch.stack([xVis[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
            isFtrsTx = torch.stack([xTx[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsAud = torch.zeros_like(isFtrsVis).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis > 0 and numSegsAud > 0 and numSegsTx == 0:
            xVis, xAud = x[:numNodesInMod, :], x[numNodesInMod: 2 * numNodesInMod, :]
            isFtrsVis = torch.stack([xVis[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
            isFtrsAud = torch.stack([xAud[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsTx = torch.zeros_like(isFtrsVis).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis == 0 and numSegsAud == 0 and numSegsTx > 0:
            isFtrsTx = torch.stack([x[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsVis = torch.zeros_like(isFtrsTx).to(device)
            isFtrsAud = torch.zeros_like(isFtrsTx).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx == 0:
            isFtrsAud = torch.stack([x[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsVis = torch.zeros_like(isFtrsAud).to(device)
            isFtrsTx = torch.zeros_like(isFtrsAud).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx == 0:
            isFtrsVis = torch.stack([x[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsAud = torch.zeros_like(isFtrsVis).to(device)
            isFtrsTx = torch.zeros_like(isFtrsVis).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        else:
            raise Exception('All input features are empty.')
        return isFtrs

class isGNNNorm(torch.nn.Module):
    def __init__(self, inChl, hdChlList, outChl, numHeads):
        super(isGNNNorm, self).__init__()
        gatHdChlList = [int(hdChl/numHeads) for hdChl in hdChlList]
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChlList[0], heads=numHeads)
        self.norm1 = nn.LayerNorm(hdChlList[0])
        self.conv2 = GATv2Conv(in_channels=hdChlList[0], out_channels=gatHdChlList[1], heads=numHeads)
        self.norm2 = nn.LayerNorm(hdChlList[1])
        self.conv3 = GATv2Conv(in_channels=hdChlList[1], out_channels=gatHdChlList[2], heads=numHeads)
        self.norm3 = nn.LayerNorm(hdChlList[2])
        self.conv4 = GATv2Conv(in_channels=hdChlList[2], out_channels=gatOutChl, heads=numHeads)
        self.norm4 = nn.LayerNorm(outChl)
        self.dropout = nn.Dropout()
    def forward(self, x, edgeIds, numSegsVis, numSegsAud, numSegsTx, numNodesInMod, istnsIds, device):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x, edgeIds)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x, edgeIds)
        x = self.norm3(x)
        x = F.relu(x)

        x = self.conv4(x, edgeIds)
        x = self.norm4(x)
        x = F.relu(x)
        if numSegsVis > 0 and numSegsAud > 0 and numSegsTx > 0:
             xVis, xAud, xTx = x[:numNodesInMod, :], x[numNodesInMod : 2 * numNodesInMod, :], x[2 * numNodesInMod:, :]
             isFtrsVis = torch.stack([xVis[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
             isFtrsAud = torch.stack([xAud[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
             isFtrsTx = torch.stack([xTx[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
             isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx > 0:
            xAud, xTx = x[:numNodesInMod, :], x[numNodesInMod: 2 * numNodesInMod, :]
            isFtrsAud = torch.stack([xAud[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
            isFtrsTx = torch.stack([xTx[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsVis = torch.zeros_like(isFtrsAud).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx > 0:
            xVis, xTx = x[:numNodesInMod, :], x[numNodesInMod: 2 * numNodesInMod, :]
            isFtrsVis = torch.stack([xVis[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
            isFtrsTx = torch.stack([xTx[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsAud = torch.zeros_like(isFtrsVis).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis > 0 and numSegsAud > 0 and numSegsTx == 0:
            xVis, xAud = x[:numNodesInMod, :], x[numNodesInMod: 2 * numNodesInMod, :]
            isFtrsVis = torch.stack([xVis[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)
            isFtrsAud = torch.stack([xAud[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsTx = torch.zeros_like(isFtrsVis).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis == 0 and numSegsAud == 0 and numSegsTx > 0:
            isFtrsTx = torch.stack([x[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsVis = torch.zeros_like(isFtrsTx).to(device)
            isFtrsAud = torch.zeros_like(isFtrsTx).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx == 0:
            isFtrsAud = torch.stack([x[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsVis = torch.zeros_like(isFtrsAud).to(device)
            isFtrsTx = torch.zeros_like(isFtrsAud).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx == 0:
            isFtrsVis = torch.stack([x[istnsIds[i]: istnsIds[i + 1]].mean(dim=0) for i in range(istnsIds.shape[0] - 1)], dim=0)

            isFtrsAud = torch.zeros_like(isFtrsVis).to(device)
            isFtrsTx = torch.zeros_like(isFtrsVis).to(device)
            isFtrs = torch.cat((isFtrsVis, isFtrsAud, isFtrsTx), dim=1)
        else:
            raise Exception('All input features are empty.')
        return isFtrs


class GAN2L(torch.nn.Module):
    def __init__(self, inChl, hdChl, outChl, numCls, numHeads):
        super(GAN2L, self).__init__()
        gatHdChl = int(hdChl/numHeads)
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChl, heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChl, out_channels=gatOutChl, heads=numHeads)
        self.clf = nn.Linear(outChl, numCls)
    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds)
        x = torch.unsqueeze(x,dim=0)
        #x = torch.mean(x, 1)
        x = fuse_average(x)
        x = self.clf(x)
        return x

class GAN4L(torch.nn.Module):
    def __init__(self, inChl, hdChl, outChl, numCls, numHeads):
        super(GAN4L, self).__init__()
        gatHdChl = int(hdChl/numHeads)
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChl, heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv3 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv4 = GATv2Conv(in_channels=hdChl, out_channels=gatOutChl, heads=numHeads)
        self.clf = nn.Linear(outChl, numCls)
    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds).relu()
        x = self.conv3(x, edgeIds).relu()
        x = self.conv4(x, edgeIds)
        x = torch.unsqueeze(x,dim=0)
        #x = torch.sum(x, 1)
        x = fuse_average(x)
        x = self.clf(x)
        return x

class GAN8L(torch.nn.Module):
    def __init__(self, inChl, hdChl, outChl, numCls, numHeads):
        super(GAN8L, self).__init__()
        gatHdChl = int(hdChl/numHeads)
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChl, heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv3 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv4 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv5 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv6 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv7 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv8 = GATv2Conv(in_channels=hdChl, out_channels=gatOutChl, heads=numHeads)
        self.clf = nn.Linear(outChl, numCls)
    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds).relu()
        x = self.conv3(x, edgeIds).relu()
        x = self.conv4(x, edgeIds).relu()
        x = self.conv5(x, edgeIds).relu()
        x = self.conv6(x, edgeIds).relu()
        x = self.conv7(x, edgeIds).relu()
        x = self.conv8(x, edgeIds)
        x = torch.unsqueeze(x,dim=0)
        #x = torch.mean(x, 1)
        x = fuse_average(x)
        x = self.clf(x)
        return x

#Average-1/3
def fuse_average(ftrs):
    ftrVis, ftrAud, ftrTx = torch.mean(ftrs[:,:100], 1), torch.mean(ftrs[:,100:200], 1), ftrs[:,200]
    avgFtr = (ftrVis + ftrAud + ftrTx) / 3
    return avgFtr
#Average-1/2
def fuse_average2(ftrs):
    ftrVis, ftrAud = torch.mean(ftrs[:,:100], 1), torch.mean(ftrs[:,100:200], 1)
    avgFtr = (ftrVis + ftrAud) / 2
    return avgFtr
#sum 1/3
def fuse_sum(ftrs):
    ftrVis, ftrAud, ftrTx = torch.mean(ftrs[:,:100], 1), torch.mean(ftrs[:,100:200], 1), ftrs[:,200]
    avgFtr = ftrVis + ftrAud + ftrTx
    return avgFtr

class GAN4LS(torch.nn.Module):
    def __init__(self, inChl, hdChl, outChl, numCls, numHeads):
        super(GAN4LS, self).__init__()
        gatHdChl = int(hdChl/numHeads)
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChl, heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv3 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv4 = GATv2Conv(in_channels=hdChl, out_channels=gatOutChl, heads=numHeads)
        self.clf = nn.Linear(outChl, numCls)
    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds).relu()
        x = self.conv3(x, edgeIds).relu()
        x = self.conv4(x, edgeIds)
        x = torch.unsqueeze(x,dim=0)
        x = torch.mean(x, 1)
        x = self.clf(x)
        return x

class GAN4LWt(torch.nn.Module):
    def __init__(self, inChl, hdChl, outChl, numCls, numHeads):
        super(GAN4LWt, self).__init__()
        gatHdChl = int(hdChl/numHeads)
        gatOutChl = int(outChl/numHeads)
        self.conv1 = GATv2Conv(in_channels=inChl, out_channels=gatHdChl, heads=numHeads)
        self.conv2 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv3 = GATv2Conv(in_channels=hdChl, out_channels=gatHdChl, heads=numHeads)
        self.conv4 = GATv2Conv(in_channels=hdChl, out_channels=gatOutChl, heads=numHeads)
        self.clf = nn.Linear(outChl, numCls)

    def forward(self, x, edgeIds):
        x = torch.squeeze(x)
        x = self.conv1(x, edgeIds).relu()
        x = self.conv2(x, edgeIds).relu()
        x = self.conv3(x, edgeIds).relu()
        x = self.conv4(x, edgeIds)
        x = torch.unsqueeze(x,dim=0)
        return x
    def cls(self, ftrs, wt):
        ftrWr = ftrs * wt
        ftrCls = fuse_average(ftrWr)
        output = self.clf(ftrCls)
        return output


