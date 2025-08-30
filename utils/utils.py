from os import supports_bytes_environ

import torch
import numpy as np
import random
from sklearn.metrics import *
import torch_geometric
import torch.nn.functional as F

def fix_the_random(seed_val = 2021):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch_geometric.seed_everything(seed_val)


def evalMetric(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_true))
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
        recallScore = recall_score(y_true, y_pred, labels = np.unique(y_true))
        precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_true))
    except:
        return dict({"accuracy": 0, 'mF1Score': 0, 'f1Score': 0, 'auc': 0,'precision': 0, 'recall': 0})
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
           'precision': precisionScore, 'recall': recallScore})

def cal_edge_ids_intra_adj_and_inter_3(numSegs, device):
    nodesVis = torch.arange(numSegs).to(device)
    edgeIdsIntraAdjVis = torch.vstack((nodesVis[:-1], nodesVis[1:])) #[2, numSeg-1]
    edgeIdsIntraAdjAud = edgeIdsIntraAdjVis + numSegs #[2, numSeg-1]
    edgeIdsIntraAdjTx = edgeIdsIntraAdjVis + 2 * numSegs #[2, numSeg-1]
    edgeIdsIntraAdj = torch.cat((edgeIdsIntraAdjVis, edgeIdsIntraAdjAud, edgeIdsIntraAdjTx), dim=1) #[2, 3*(numSeg-1)]

    edgeIdsInterVisAud = torch.vstack((nodesVis, nodesVis + numSegs)) #[2, numSeg]
    edgeIdsInterVisTx = torch.vstack((nodesVis, nodesVis + 2 * numSegs)) #[2, numSeg]
    edgeIdsInterAudTx = torch.vstack((nodesVis + numSegs, nodesVis + 2 * numSegs)) #[2, numSeg]
    edgeIdsInter = torch.cat((edgeIdsInterVisAud, edgeIdsInterVisTx, edgeIdsInterAudTx), dim=1) #[2, 3 * numSeg]

    return edgeIdsIntraAdj, edgeIdsInter

def cal_edge_ids_intra_adj_and_inter_2(numSegs, device):
    nodesM1 = torch.arange(numSegs).to(device)
    edgeIdsIntraAdjM1 = torch.vstack((nodesM1[:-1], nodesM1[1:])) #[2, numSeg-1]
    edgeIdsIntraAdjM2 = edgeIdsIntraAdjM1 + numSegs #[2, numSeg-1]
    edgeIdsIntraAdj = torch.cat((edgeIdsIntraAdjM1, edgeIdsIntraAdjM2), dim=1) #[2, 2*(numSeg-1)]

    edgeIdsInter = torch.vstack((nodesM1, nodesM1 + numSegs)) #[2, numSeg]

    return edgeIdsIntraAdj, edgeIdsInter

def cal_edge_ids_intra_adj_1(numSegs, device):
    nodes = torch.arange(numSegs).to(device)
    edgeIdsIntraAdj = torch.vstack((nodes[:-1], nodes[1:])) #[2, numSeg-1]

    return edgeIdsIntraAdj

def cal_edge_ids_intra_CS(nodeFtrs, numSegs, epsilon, device):
    nodeFtrsN = F.normalize(nodeFtrs, p=2, dim=1)
    cosSim = nodeFtrsN @ nodeFtrsN.t()
    cosDist = 1.0 - cosSim

    r_tri, c_tri = torch.triu_indices(numSegs, numSegs, offset=1, device=device)
    keep = cosDist[r_tri, c_tri] < epsilon
    r = r_tri[keep]
    c = c_tri[keep]
    edgeIdsIntraCs = torch.stack((r, c), dim=0)

    return edgeIdsIntraCs

def handle_3(nodeFtrVis, nodeFtrAud, nodeFtrTx, epsilon, numSegs, device):
    edgeIdsIntraAdj, edgeIdsInter = cal_edge_ids_intra_adj_and_inter_3(numSegs, device)
    edgeIdsIntraCSVis = cal_edge_ids_intra_CS(nodeFtrVis, numSegs, epsilon, device)
    edgeIdsIntraCSAud = cal_edge_ids_intra_CS(nodeFtrAud, numSegs, epsilon, device) + numSegs
    edgeIdsIntraCSTx = cal_edge_ids_intra_CS(nodeFtrTx, numSegs, epsilon, device) + 2 * numSegs

    edgeIdsPart1 = torch.cat((edgeIdsIntraAdj, edgeIdsIntraCSVis, edgeIdsIntraCSAud, edgeIdsIntraCSTx, edgeIdsInter), dim=1)
    # edgeIdsPart2 = torch.stack((edgeIdsPart1[1,:], edgeIdsPart1[0,:]), dim=0)
    # edgeIds = torch.cat((edgeIdsPart1, edgeIdsPart2), dim=1)
    return edgeIdsPart1

def handle_2(nodeFtrM1, nodeFtrM2, epsilon, numSegs, device):
    edgeIdsIntraAdj, edgeIdsInter = cal_edge_ids_intra_adj_and_inter_2(numSegs, device)
    edgeIdsIntraCSM1 = cal_edge_ids_intra_CS(nodeFtrM1, numSegs, epsilon, device)
    edgeIdsIntraCSM2 = cal_edge_ids_intra_CS(nodeFtrM2, numSegs, epsilon, device) + numSegs

    edgeIdsPart1 = torch.cat((edgeIdsIntraAdj, edgeIdsIntraCSM1, edgeIdsIntraCSM2, edgeIdsInter), dim=1)
    # edgeIdsPart2 = torch.stack((edgeIdsPart1[1, :], edgeIdsPart1[0, :]), dim=0)
    # edgeIds = torch.cat((edgeIdsPart1, edgeIdsPart2), dim=1)
    return edgeIdsPart1

def handle_1(nodeFtr, epsilon, numSegs, device):
    edgeIdsIntraAdj = cal_edge_ids_intra_adj_1(numSegs, device)
    edgeIdsIntraCS = cal_edge_ids_intra_CS(nodeFtr, numSegs, epsilon, device)

    edgeIdsPart1 = torch.cat((edgeIdsIntraAdj, edgeIdsIntraCS), dim=1)
    # edgeIdsPart2 = torch.stack((edgeIdsPart1[1, :], edgeIdsPart1[0, :]), dim=0)
    # edgeIds = torch.cat((edgeIdsPart1, edgeIdsPart2), dim=1)
    return edgeIdsPart1

# def cal_edge_ids(nodeFtrVis, nodeFtrAud, nodeFtrTx, epsilon, device):
#     # nodeFtr with shape: [N, D], where N is the number of segments, D is the dim of features.
#     numSegsVis, numSegsAud, numSegsTx = nodeFtrVis.shape[0], nodeFtrAud.shape[0], nodeFtrTx.shape[0]
#
#     if numSegsVis > 0 and numSegsAud > 0 and numSegsTx > 0:
#         edgeIds = handle_3(nodeFtrVis, nodeFtrAud, nodeFtrTx, epsilon, numSegsVis, device)
#         numMod, numNodesInMod = 3, numSegsVis
#     elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx > 0:
#         edgeIds = handle_2(nodeFtrAud, nodeFtrTx, epsilon, numSegsAud, device)
#         numMod, numNodesInMod = 2, numSegsAud
#     elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx > 0:
#         edgeIds = handle_2(nodeFtrVis, nodeFtrTx, epsilon, numSegsVis, device)
#         numMod, numNodesInMod = 2, numSegsVis
#     elif numSegsVis > 0 and numSegsAud > 0 and numSegsTx == 0:
#         edgeIds = handle_2(nodeFtrVis, nodeFtrAud, epsilon, numSegsVis, device)
#         numMod, numNodesInMod = 2, numSegsVis
#     elif numSegsVis == 0 and numSegsAud == 0 and numSegsTx > 0:
#         edgeIds = handle_1(nodeFtrTx, epsilon, numSegsTx, device)
#         numMod, numNodesInMod = 1, numSegsTx
#     elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx == 0:
#         edgeIds = handle_1(nodeFtrAud, epsilon, numSegsAud, device)
#         numMod, numNodesInMod = 1, numSegsAud
#     elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx == 0:
#         edgeIds = handle_1(nodeFtrVis, epsilon, numSegsVis, device)
#         numMod, numNodesInMod = 1, numSegsVis
#     else:
#         raise Exception('All input features are empty.')
#
#     edgeIds = edgeIds.to(torch.int64)
#
#     return edgeIds, numMod, numNodesInMod

def get_nodes_and_edges(ftrVis, ftrAud, ftrTx, modelVis, modelAud, modelTx, epsilon, istnsIds, device):

    numSegsVis, numSegsAud, numSegsTx = ftrVis.shape[1], ftrAud.shape[1], ftrTx.shape[1]
    if numSegsVis > 0 and numSegsAud > 0 and numSegsTx > 0:
        nodeFtrVis, nodeFtrAud, nodeFtrTx = modelVis(ftrVis), modelAud(ftrAud), modelTx(ftrTx)
        nodeFtrs = torch.cat((nodeFtrVis, nodeFtrAud, nodeFtrTx), 1)
        edgeIdsPart1WtGNN = handle_3(nodeFtrVis.squeeze(), nodeFtrAud.squeeze(), nodeFtrTx.squeeze(), epsilon, numSegsVis, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 3, numSegsVis

        subGraphLabelsTemp = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabelsTemp[istnsIds[k]: istnsIds[k + 1]] = k
        subGraphLabels = torch.cat((subGraphLabelsTemp, subGraphLabelsTemp, subGraphLabelsTemp))
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]

    elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx > 0:
        nodeFtrAud, nodeFtrTx = modelAud(ftrAud), modelTx(ftrTx)
        nodeFtrs = torch.cat((nodeFtrAud, nodeFtrTx), 1)
        edgeIdsPart1WtGNN = handle_2(nodeFtrAud.squeeze(), nodeFtrTx.squeeze(), epsilon, numSegsAud, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 2, numSegsAud

        subGraphLabelsTemp = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabelsTemp[istnsIds[k]: istnsIds[k + 1]] = k
        subGraphLabels = torch.cat((subGraphLabelsTemp, subGraphLabelsTemp))
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]

    elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx > 0:
        nodeFtrVis, nodeFtrTx = modelVis(ftrVis), modelTx(ftrTx)
        nodeFtrs = torch.cat((nodeFtrVis, nodeFtrTx), 1)
        edgeIdsPart1WtGNN = handle_2(nodeFtrVis.squeeze(), nodeFtrTx.squeeze(), epsilon, numSegsVis, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 2, numSegsVis

        subGraphLabelsTemp = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabelsTemp[istnsIds[k]: istnsIds[k + 1]] = k
        subGraphLabels = torch.cat((subGraphLabelsTemp, subGraphLabelsTemp))
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]

    elif numSegsVis > 0 and numSegsAud > 0 and numSegsTx == 0:
        nodeFtrVis, nodeFtrAud = modelVis(ftrVis), modelAud(ftrAud)
        nodeFtrs = torch.cat((nodeFtrVis, nodeFtrAud), 1)
        edgeIdsPart1WtGNN = handle_2(nodeFtrVis.squeeze(), nodeFtrAud.squeeze(), epsilon, numSegsVis, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 2, numSegsVis

        subGraphLabelsTemp = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabelsTemp[istnsIds[k]: istnsIds[k + 1]] = k
        subGraphLabels = torch.cat((subGraphLabelsTemp, subGraphLabelsTemp))
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]

    elif numSegsVis == 0 and numSegsAud == 0 and numSegsTx > 0:
        nodeFtrTx = modelTx(ftrTx)
        nodeFtrs = nodeFtrTx
        edgeIdsPart1WtGNN = handle_1(nodeFtrTx.squeeze(), epsilon, numSegsTx, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 1, numSegsTx

        subGraphLabels = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabels[istnsIds[k]: istnsIds[k + 1]] = k
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]

    elif numSegsVis == 0 and numSegsAud > 0 and numSegsTx == 0:
        nodeFtrAud = modelAud(ftrAud)
        nodeFtrs = nodeFtrAud
        edgeIdsPart1WtGNN = handle_1(nodeFtrAud.squeeze(), epsilon, numSegsAud, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 1, numSegsAud

        subGraphLabels = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabels[istnsIds[k]: istnsIds[k + 1]] = k
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]

    elif numSegsVis > 0 and numSegsAud == 0 and numSegsTx == 0:
        nodeFtrVis = modelVis(ftrVis)
        nodeFtrs = nodeFtrVis
        edgeIdsPart1WtGNN = handle_1(nodeFtrVis.squeeze(), epsilon, numSegsVis, device)
        edgeIdsPart1WtGNN =  edgeIdsPart1WtGNN.to(torch.int64)
        numMod, numNodesInMod = 1, numSegsVis

        subGraphLabels = torch.empty(numNodesInMod).to(device)
        for k in range(istnsIds.shape[0] - 1):
            subGraphLabels[istnsIds[k]: istnsIds[k + 1]] = k
        u, v = edgeIdsPart1WtGNN
        same = (subGraphLabels[u] == subGraphLabels[v])
        edgeIdsPart1IsGNN = edgeIdsPart1WtGNN[:, same]
    else:
        raise Exception('All input features are empty.')

    edgeIdsPart2WtGNN = torch.stack((edgeIdsPart1WtGNN[1,:], edgeIdsPart1WtGNN[0,:]), dim=0)
    edgeIdsWtGNN = torch.cat((edgeIdsPart1WtGNN, edgeIdsPart2WtGNN), dim=1)
    edgeIdsPart2IsGNN = torch.stack((edgeIdsPart1IsGNN[1,:], edgeIdsPart1IsGNN[0,:]), dim=0)
    edgeIdsIsGNN = torch.cat((edgeIdsPart1IsGNN, edgeIdsPart2IsGNN), dim=1)

    return nodeFtrs, edgeIdsWtGNN, edgeIdsIsGNN, numMod, numNodesInMod

def get_instance_ids(numSegs, device):
    if numSegs == 0:
        raise Exception('All input features are empty.')
    else:
        numIstns = min(numSegs, 10)

    numNodesInIstn = numSegs // numIstns
    rest = numSegs % numIstns
    numNodesInIstnAllMain = torch.ones(numIstns).to(device) * numNodesInIstn
    numNodesInIstnAllSupp = torch.cat((torch.ones(rest).to(device), torch.zeros(numIstns-rest).to(device)), dim=0)
    numNodesInIstnAll = numNodesInIstnAllMain + numNodesInIstnAllSupp
    istnsIdsMain = torch.cumsum(numNodesInIstnAll, dim=0).to(torch.int64)
    istnsIds = torch.cat((torch.zeros(1).to(device), istnsIdsMain), dim=0).to(torch.int64)
    return istnsIds





