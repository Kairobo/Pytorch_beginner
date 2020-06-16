import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='CustomNet')

    # for data
    parser.add_argument('--customDataFolder', dest='customDataFolder',
                        help='data folder',
                        default='./data/data', type=str)

    # for model choice
    parser.add_argument('--inputResolution', dest='inputResolution',
                        help='the input resolution for the network',
                        default=256, type=int)
    parser.add_argument('--loadModel', dest='loadModel',
                        help='load model',
                        default=False, type=bool)



    # for training
    parser.add_argument('--useDataAug', dest='useDataAug',
                        help='use data augmentation',
                        default=False, type=bool)
    parser.add_argument('--useCuda', dest='useCuda',
                        help='use cuda',
                        default=False, type=bool)
    parser.add_argument('--randomSeed', dest='randomSeed',
                        help='the random seed for pytorch',
                        default=1, type=int)
    parser.add_argument('--logInterval', dest='logInterval',
                        help='the log interval when training',
                        default=1, type=int)
    parser.add_argument('--logValInterval', dest='logValInterval',
                        help='the log interval for validation when training',
                        default=1, type=int)
    parser.add_argument('--modelSaveInterval', dest='modelSaveInterval',
                        help='the interval for saving model weight when training',
                        default=500, type=int)
    parser.add_argument('--numEpochs', dest='numEpochs',
                        help='the number of epochs',
                        default=3, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=4, type=int)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--Momentum', dest='Momentum',
                        help='Momentum for learning',
                        default=0.5, type=float)


    args = parser.parse_args()
    return args