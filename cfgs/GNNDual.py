import argparse
import datetime
import os

def parse_args():
    cur_dir = os.getcwd().replace('\\', '/')
    prnt_dir = os.path.abspath(os.path.join(cur_dir, os.pardir)).replace('\\', '/')
    now = str(datetime.datetime.now())[:19].replace(':', '-').replace(' ', '_')

    # train
    parser = argparse.ArgumentParser(description='Basic GNN')
    parser.add_argument('--logDir', default=prnt_dir + '/logs/' + now)
    parser.add_argument('--saveDir', default=prnt_dir + '/models/saved_models')
    parser.add_argument('--gpuIndex', '-gi', type=int, default=0)
    parser.add_argument('--trnSaveFile', type=str, default='/trn_attr.pt')
    parser.add_argument('--tsSaveFile', type=str, default='/ts_attr.pt')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--batchSz', default=1)
    parser.add_argument('--numEps', default=40)
    parser.add_argument('--numWks', default=0) ### or 2
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--pm', default=True)
    parser.add_argument('--epsl', default=0.2)
    # parser.add_argument('--epsl', default=0.1)
    # parser.add_argument('--epsl', default=0.3)


    # model
    parser.add_argument('--lstmD', default=128)
    parser.add_argument('--numLay', default=4)
    parser.add_argument('--nodeFtrD', default=32)
    parser.add_argument('--numHeads', default=2)
    parser.add_argument('--numCls', default=2)
    parser.add_argument('--wtGNNHdChlList', default=[32,64,64])
    parser.add_argument('--isGNNHdChlList', default=[32,64,64])
    parser.add_argument('--mlpNdSrsHdSz', default=[64, 64])
    parser.add_argument('--mlpClsHdSz', default=[256, 64])



    # data
    parser.add_argument('--visEmdSz', default=768)
    parser.add_argument('--audEmdSz', default=40)
    parser.add_argument('--txEmdSz', default=768)
    parser.add_argument('--txHdSz', default=[128, 128])

    args=parser.parse_args()
    return args
