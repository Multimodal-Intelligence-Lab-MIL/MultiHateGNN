import os

curPath = os.getcwd().replace('\\', '/')
prntPath =os.path.abspath(os.path.join(curPath, os.pardir)).replace('\\', '/')
os.sys.path.append(prntPath)

import pickle
import torch
from cfgs.GNNDual import parse_args
from torch.utils.tensorboard import SummaryWriter
from dataset import hateMM
from torch.utils.data import DataLoader
from utils.utils import *
from models.GNN import *
from models.models_tran_feature import *
import torch.optim as optim

# fix_the_random(2024)


def load_data():
    visFtrAllPath = prntPath + '/dataset/hateMM/visFtrAll.pkl'
    with open(visFtrAllPath, 'rb') as f:
        visFtrAll = pickle.load(f)
    audFtrAllPath = prntPath + '/dataset/hateMM/audFtrAll.pkl'
    with open(audFtrAllPath, 'rb') as f:
        audFtrAll = pickle.load(f)
    textFtrAllPath = prntPath + '/dataset/hateMM/textFtrAll.pkl'
    with open(textFtrAllPath, 'rb') as f:
        textFtrAll = pickle.load(f)
    return visFtrAll, audFtrAll, textFtrAll

def train(loader):
    modelVis.train()
    modelAud.train()
    modelTx.train()
    modelWtGNN.train()
    modelIsGNN.train()
    modelIsWts.train()
    modelCls.train()


    totLoss = 0
    all_y_pred = []
    all_y_true = []
    N_count = 0  # counting total trained sample in one epoch

    for batchIdx, (ftrVis, ftrAud, ftrTx, y) in enumerate(loader):
        ftrVis, ftrAud, ftrTx, y = (ftrVis.float()).to(device), (ftrAud.float()).to(device), (ftrTx.float()).to(device), y.to(device).view(-1)
        N_count += args.batchSz
        numSegsVis, numSegsAud, numSegsTx = ftrVis.shape[1], ftrAud.shape[1], ftrTx.shape[1]
        numSegs = max(numSegsVis, numSegsAud, numSegsTx)
        istnsIds = get_instance_ids(numSegs, device)
        nodeFtrs, edgeIdsWtGNN, edgeIdsIsGNN, numMod, numNodesInMod = get_nodes_and_edges(ftrVis, ftrAud, ftrTx, modelVis, modelAud, modelTx, args.epsl, istnsIds, device)

        outputsWt = modelWtGNN(nodeFtrs, edgeIdsWtGNN) # outputsWt: [batch, number of nodes, dim of node features]
        isWts = modelIsWts(outputsWt, numMod, numNodesInMod, istnsIds)
        isFtrs = modelIsGNN(nodeFtrs, edgeIdsIsGNN, numSegsVis, numSegsAud, numSegsTx, numNodesInMod, istnsIds, device)
        output = modelCls(isWts, isFtrs)
        loss = F.cross_entropy(output, y, weight=torch.FloatTensor([0.41, 0.59]).to(device)) #output: (batch, number of classes); y: [batch]
        # loss = F.cross_entropy(output, y, weight=torch.FloatTensor([0.2, 0.8]).to(
        #     device))  # output: (batch, number of classes); y: [batch]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totLoss += loss

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        all_y_pred.extend(y_pred.cpu().numpy())
        all_y_true.extend(y.cpu().numpy())
    metrics = evalMetric(np.array(all_y_true), np.array(all_y_pred))
    avgLoss = totLoss/N_count

    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tavgLoss: {:.6f}, Accu: {:.2f}%, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(
    #             ep + 1, N_count, len(trnLoader.dataset), 100. * (batchIdx + 1) / len(trnLoader), avgLoss.item(),
    #             100 * metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'],
    #             metrics['precision'], metrics['recall']))

    return avgLoss, metrics
def eval(loader):
    modelVis.eval()
    modelAud.eval()
    modelTx.eval()
    modelWtGNN.eval()
    modelIsGNN.eval()
    modelIsWts.eval()
    modelCls.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    all_y_predSlt_1, all_y_predSlt_2, all_y_predSlt_3, all_y_predSlt_5, all_y_predSlt_7 = [], [], [], [], []
    with torch.no_grad():
        for batchIdx, (ftrVis, ftrAud, ftrTx, y) in enumerate(loader):
            ftrVis, ftrAud, ftrTx, y = (ftrVis.float()).to(device), (ftrAud.float()).to(device), (ftrTx.float()).to(
                device), y.to(device).view(-1)
            numSegsVis, numSegsAud, numSegsTx = ftrVis.shape[1], ftrAud.shape[1], ftrTx.shape[1]
            numSegs = max(numSegsVis, numSegsAud, numSegsTx)
            istnsIds = get_instance_ids(numSegs, device)
            nodeFtrs, edgeIdsWtGNN, edgeIdsIsGNN, numMod, numNodesInMod = get_nodes_and_edges(ftrVis, ftrAud, ftrTx,
                                                                                              modelVis, modelAud,
                                                                                              modelTx, args.epsl,
                                                                                              istnsIds, device)
            outputsWt = modelWtGNN(nodeFtrs, edgeIdsWtGNN)  # outputsWt: [batch, number of nodes, dim of node features]
            isWts = modelIsWts(outputsWt, numMod, numNodesInMod, istnsIds)
            isFtrs = modelIsGNN(nodeFtrs, edgeIdsIsGNN, numSegsVis, numSegsAud, numSegsTx, numNodesInMod, istnsIds, device)

            isWtsSt, isWtsIds = torch.sort(isWts)

            output = modelCls(isWts, isFtrs)


            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()  # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)



    test_loss /= len(evalLoader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)


    print("---------------------")

    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())




    return test_loss, metrics



args = parse_args()
logger = SummaryWriter(args.logDir)
# device = torch.device('cuda', index=args.gpuIndex) if torch.cuda.is_available() else torch.device('cpu')
# if torch.cuda.is_available():
#     torch.cuda.set_device(args.gpuIndex)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
args.device = device

#load features from pretrained models
visFtrAll, audFtrAll, textFtrAll = load_data()

allFd = ['Fold_2', 'Fold_2', 'Fold_2', 'Fold_2', 'Fold_2']
fnlAccFds = []
for fd in allFd:
    saveName = args.saveDir + fd + '.pt'
    sltType = fd
    trnSet = hateMM(sltType, 'trn', visFtrAll, audFtrAll, textFtrAll)
    valSet = hateMM(sltType, 'val', visFtrAll, audFtrAll, textFtrAll)
    evalSet = hateMM(sltType, 'ts', visFtrAll, audFtrAll, textFtrAll)
    trnLoader = DataLoader(trnSet, batch_size= args.batchSz, shuffle=args.shuffle, num_workers=args.numWks, pin_memory=args.pm)
    valLoader = DataLoader(valSet, batch_size= args.batchSz, shuffle=False, num_workers=args.numWks, pin_memory=args.pm)
    evalLoader = DataLoader(evalSet, batch_size= args.batchSz, shuffle=False, num_workers=args.numWks, pin_memory=args.pm)

    modelVis = LSTM(args.lstmD, args.visEmdSz, args.nodeFtrD).to(args.device)
    modelAud = LSTM(args.lstmD, args.audEmdSz, args.nodeFtrD).to(args.device)
    modelTx = Text_Model(args.txEmdSz, args.txHdSz, args.nodeFtrD).to(args.device)
    modelWtGNN = wtGNN(args.nodeFtrD, args.wtGNNHdChlList, args.nodeFtrD, args.numHeads).to(args.device)
    modelIsGNN = isGNN(args.nodeFtrD, args.isGNNHdChlList, args.nodeFtrD, args.numHeads).to(args.device)
    modelIsWts = wtMLP(args.nodeFtrD, args.mlpNdSrsHdSz, 1).to(args.device).to(args.device)
    modelCls = classifier(args.nodeFtrD * 3, args.mlpClsHdSz, args.numCls).to(args.device)

    # optimizer = optim.Adam([{'params': modelVis.parameters()}, {'params': modelAud.parameters()}, {'params': modelTx.parameters()},
    #                         {'params': modelWtGNN.parameters()}, {'params': modelIsGNN.parameters()}, {'params': modelIsWts.parameters()},
    #                         {'params': modelCls.parameters()}], lr=args.lr)
    optimizer = optim.AdamW([{'params': modelVis.parameters()}, {'params': modelAud.parameters()}, {'params': modelTx.parameters()},
                            {'params': modelWtGNN.parameters()}, {'params': modelIsGNN.parameters()}, {'params': modelIsWts.parameters()},
                            {'params': modelCls.parameters()}], lr=args.lr, weight_decay=0.1)


    trnLossAll, trnAccAll, evalLossAll, evalAccAll = [], [], [], []
    bestAcc = 0
    bestTsAccSlt_1, bestTsAccSlt_2, bestTsAccSlt_3, bestTsAccSlt_5, bestTsAccSlt_7 = 0, 0, 0, 0, 0
    evalFnlVal = None
    for ep in range(args.numEps):
        trnAvgLoss, trnMtc = train(trnLoader)
        valLoss, valMtc = eval(valLoader)
        evalLoss, evalMtc = eval(evalLoader)
        if valMtc['accuracy'] > bestAcc:
            bestAcc = evalMtc['accuracy']

        print('epoch:', ep, 'trnAcc:', trnMtc['accuracy'], 'bestAcc:', bestAcc)

        print()

    fnlAccFds.append(bestAcc)
