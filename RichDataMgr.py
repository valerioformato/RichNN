import numpy as np
import uproot as up

from keras.utils import Sequence

class RichDataMgr(Sequence):
    def __init__(self, rootfile = None, batch_size = 16, is_training=True):
        self.rootfile = rootfile
        self.tree = None
        self.is_training = is_training
        self.testing_fraction = 0.2
        self.firstev = 0
        self.batch_size = batch_size
        self.dim = 0
        self.pad = 0
        self.branches = [b'RichHits', b'RichHits.xPix', b'RichHits.yPix', b'RichHits.nPhElUncorr', b'RichHits.status']

        if self.rootfile:
            self.Open(self.rootfile)

    def __len__(self):
        if(self.is_training):
            self.firstev = 0
            return np.ceil((1-self.testing_fraction)*len(self.tree[b'nTrTrack']) / float(self.batch_size))
        else:
            self.firstev = np.ceil((1-self.testing_fraction)*len(self.tree[b'nTrTrack']))
            return np.ceil(self.testing_fraction*len(self.tree[b'nTrTrack']) / float(self.batch_size))

    def __getitem__(self, idx):
        # print(self.firstev + idx * self.batch_size, self.firstev + (idx+1) * self.batch_size)
        # print self.tree.arrays([b'RichHits'], dict, 0, 16)#, self.firstev + 1 + idx * self.batch_size , self.firstev + 1 + (idx + 1) * self.batch_size)
        return self.ConvertFromRootFormat(self.tree.arrays(self.branches, dict, self.firstev + idx * self.batch_size , self.firstev + (idx + 1) * self.batch_size))

    def Open(self, rootfile):
        self.rootfile = up.open(rootfile)
        self.LoadTrainingTree()

    def SetShape(self, dim, pad):
        self.dim = dim
        self.pad = pad

    def LoadTrainingTree(self):
        if not self.rootfile:
            print("Error: no rootfile opened")
            return None

        if not self.tree:
            self.tree = self.rootfile['VFRichNNBuildTree/TrainingData']

    def ConvertFromRootFormat(self, arrays):
        nEv = len(arrays[b'RichHits'])
        _x, _y = [], []

        # print(nEv, arrays)
        for iEv in range(0, nEv):
            im = np.zeros((self.dim + 2*self.pad, self.dim + 2*self.pad, 1))
            target = np.zeros((self.dim + 2*self.pad, self.dim + 2*self.pad, 2))
            nHits = arrays[b'RichHits'][iEv]

            for ihit in range(0, nHits):
                xPix = arrays[b'RichHits.xPix'][iEv][ihit] + self.pad
                yPix = arrays[b'RichHits.yPix'][iEv][ihit] + self.pad

                im[yPix][xPix] = [arrays[b'RichHits.nPhElUncorr'][iEv][ihit]]

                if (arrays[b'RichHits.status'][iEv][ihit] & 1) > 0:
                    target[yPix][xPix] = [1, 1 if (arrays[b'RichHits.nPhElUncorr'][iEv][ihit] > 0) else 0]
                else:
                    target[yPix][xPix] = [0, 1 if (arrays[b'RichHits.nPhElUncorr'][iEv][ihit] > 0) else 0]

            im = np.clip(im, 0, 30)

            _x.append(im)
            _y.append(target)


        return np.array(_x), np.array(_y)



    # def GetTrainingData(self, dim, pad, frac):
    #     _x_train, _y_train, _x_test, _y_test = [], [], [], []
    #
    #     for arrays in self.TrainingTree().iterate():
    #         nEv = len(arrays[b'RichHits'])
    #
    #         nTrain = int(frac*nEv)
    #         nTest = nEv - nTrain
    #
    #         # print("RichDataMgr: Requested", nTrain, "training events and", nTest, "testing events")
    #
    #         for iEv in range(0, nEv):
    #             # im: input;
    #             #     channel #0 - phEl values
    #             # target: target values;
    #             #     channel #0 - pixel in ring or not
    #             #     channel #1 - pixel mask
    #             im = np.zeros((dim + 2*pad, dim + 2*pad, 1))
    #             target = np.zeros((dim + 2*pad, dim + 2*pad, 2))
    #             nHits = arrays[b'RichHits'][iEv]
    #             # print nHits
    #
    #             # ringHits = [[], []]
    #             # spotHits = [[], []]
    #             # trackHit = [[], []]
    #
    #             yTrPix = arrays[b'TrRichPMTIntYPix'][iEv] + pad
    #             xTrPix = arrays[b'TrRichPMTIntXPix'][iEv] + pad
    #
    #             # deltaBeta = arrays['MCBeta'][iEv] - arrays['BetaRich'][iEv]
    #
    #             for ihit in range(0, nHits):
    #                 xPix = arrays[b'RichHits.xPix'][iEv][ihit] + pad
    #                 yPix = arrays[b'RichHits.yPix'][iEv][ihit] + pad
    #
    #                 # print ihit, xPix, yPix, arrays['RichHits.nPhElUncorr'][iEv][ihit]
    #                 im[yPix][xPix] = [arrays[b'RichHits.nPhElUncorr'][iEv][ihit]]
    #
    #                 if (arrays[b'RichHits.status'][iEv][ihit] & 1) > 0:
    #                     # ringHits[0].append(xPix)
    #                     # ringHits[1].append(yPix)
    #                     target[yPix][xPix] = [1, 1 if (arrays[b'RichHits.nPhElUncorr'][iEv][ihit] > 0) else 0]
    #                 else:
    #                     target[yPix][xPix] = [0, 1 if (arrays[b'RichHits.nPhElUncorr'][iEv][ihit] > 0) else 0]
    #                 # if (arrays['RichHits.status'][iEv][ihit] & 1<<30) > 0:
    #                 #     spotHits[0].append(xPix)
    #                 #     spotHits[1].append(yPix)
    #
    #                 # trackHit[0].append(xTrPix)
    #                 # trackHit[1].append(yTrPix)
    #                 # target[yTrPix][xTrPix] = [0, 1]
    #
    #             im = np.clip(im, 0, 30)
    #
    #             if iEv < nTrain:
    #                 _x_train.append(im)
    #                 _y_train.append(target)
    #             else:
    #                 _x_test.append(im)
    #                 _y_test.append(target)
    #
    #             # imRingHits = np.array(ringHits)
    #             # imSpotHits = np.array(spotHits)
    #
    #             # fig.suptitle(r'$\beta_\mathrm{MC} - \beta_\mathrm{Rich} = ' + str(deltaBeta) + r'$')
    #             # axs[0].cla()
    #             # axs[0].imshow(im, origin='lower')
    #             # axs[0].imshow(richOcc, origin='lower', alpha=0.1)
    #             # axs[0].scatter(ringHits[0], ringHits[1], c='r', alpha=0.2)
    #             # axs[0].scatter(spotHits[0], spotHits[1], c='b', alpha=0.2)
    #             # axs[0].scatter(trackHit[0], trackHit[1], c='g', alpha=0.2)
    #             # axs[1].cla()
    #             # axs[1].imshow(target, origin='lower')
    #             # axs[1].imshow(richOcc, origin='lower', alpha=0.1)
    #             # plt.pause(1)
    #
    #     return (np.array(_x_train), np.array(_y_train)), (np.array(_x_test), np.array(_y_test))
