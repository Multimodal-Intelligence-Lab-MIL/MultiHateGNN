from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import torch


class hateMM(Dataset):
    def __init__(self, sltType, sltName, visFtrAll, audFtrAll, textFtrAll):
        super(hateMM, self).__init__()
        self.curName = os.getcwd().replace('\\', '/')
        self.prntName = os.path.abspath(os.path.join(self.curName, os.pardir)).replace('\\', '/')
        self.sltType = sltType
        self.sltName = sltName
        self.visFtrAll, self.audFtrAll, self.textFtrAll = visFtrAll, audFtrAll, textFtrAll
        self.load_data()

    def load_data(self):
        fldName = self.prntName + '/dataset/hateMM/'
        with open(fldName + 'foldsInfo.pkl', 'rb') as fp:
            allFoldDt = pickle.load(fp)
        with open(fldName + 'normInfo.pkl', 'rb') as fp:
            allNormDt = pickle.load(fp)

        self.normDt = allNormDt[self.sltType]
        if self.sltName == 'trn':
            trnList, trnLab = allFoldDt[self.sltType]['trn']
            self.idList, self.labels = trnList, trnLab
        elif self.sltName == 'val':
            valList, valLab = allFoldDt[self.sltType]['val']
            self.idList, self.labels = valList, valLab
        else:
            tsList, tsLab = allFoldDt[self.sltType]['ts']
            self.idList, self.labels = tsList, tsLab
    def read_data(self, idName):
        visFtr, audFtr, textFtr = self.visFtrAll[idName], self.audFtrAll[idName], self.textFtrAll[idName]
        if visFtr.shape[0] > 0:
            visFtr = (visFtr - self.normDt['visMean']) / self.normDt['visStd']
        if audFtr.shape[0] > 0:
            audFtr = (audFtr - self.normDt['audMean']) / self.normDt['audStd']
        if textFtr.shape[0] > 0:
            textFtr = (textFtr - self.normDt['textMean']) / self.normDt['textStd']
        return visFtr, audFtr, textFtr



    def __len__(self):
        return len(self.idList)

    def __getitem__(self, index):
        idName = self.idList[index]
        visFtr, audFtr, textFtr  = self.read_data(idName)
        y = torch.LongTensor([self.labels[index]])
        return visFtr, audFtr, textFtr, y


