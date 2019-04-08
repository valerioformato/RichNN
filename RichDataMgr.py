import numpy as np
import uproot as up

class RichDataMgr():
    def __init__(self, rootfile = None):
        self.rootfile = rootfile

        if self.rootfile:
            self.Open(self.rootfile)

    def Open(self, rootfile):
        self.rootfile = up.open('/Volumes/AMS_Disk/DBar/test/test_ML/RichNN/data/training_data.root')

    def TrainingTree(self):
        if not self.rootfile:
            print "Error: no rootfile opened"
            return None

        return self.rootfile['VFRichNNBuildTree/TrainingData']

    def GetTrainingData(self, dim, pad, frac, reqEvents=0):
        for arrays in self.TrainingTree().iterate():
            if reqEvents > 0:
                nEv = reqEvents
            else:
                nEv = len(arrays['RichHits'])

            nTrain = int(frac*nEv)
            nTest = nEv - nTrain

            # print "RichDataMgr: Requested", nTrain, "training events and", nTest, "testing events"

            _x_train, _y_train, _x_test, _y_test = [], [], [], []

            for iEv in range(0, nEv):
                # im: input;
                #     channel #0 - phEl values
                # target: target values;
                #     channel #0 - pixel in ring or not
                #     channel #1 - pixel mask
                im = np.zeros((dim + 2*pad, dim + 2*pad, 1))
                target = np.zeros((dim + 2*pad, dim + 2*pad, 2))
                nHits = arrays['RichHits'][iEv]
                # print nHits

                # ringHits = [[], []]
                # spotHits = [[], []]
                # trackHit = [[], []]

                xTrPix = arrays['TrRichPMTIntXPix'][iEv] + pad
                yTrPix = arrays['TrRichPMTIntYPix'][iEv] + pad

                # deltaBeta = arrays['MCBeta'][iEv] - arrays['BetaRich'][iEv]

                for ihit in range(0, nHits):
                    xPix = arrays['RichHits.xPix'][iEv][ihit] + pad
                    yPix = arrays['RichHits.yPix'][iEv][ihit] + pad

                    # print ihit, xPix, yPix, arrays['RichHits.nPhElUncorr'][iEv][ihit]
                    im[yPix][xPix] = [arrays['RichHits.nPhElUncorr'][iEv][ihit]]

                    if (arrays['RichHits.status'][iEv][ihit] & 1) > 0:
                        # ringHits[0].append(xPix)
                        # ringHits[1].append(yPix)
                        target[yPix][xPix] = [1, 1 if (arrays['RichHits.nPhElUncorr'][iEv][ihit] > 0) else 0]
                    else:
                        target[yPix][xPix] = [0, 1 if (arrays['RichHits.nPhElUncorr'][iEv][ihit] > 0) else 0]
                    # if (arrays['RichHits.status'][iEv][ihit] & 1<<30) > 0:
                    #     spotHits[0].append(xPix)
                    #     spotHits[1].append(yPix)

                    # trackHit[0].append(xTrPix)
                    # trackHit[1].append(yTrPix)
                    # target[yTrPix][xTrPix] = [0, 1]

                im = np.clip(im, 0, 30)

                if iEv < nTrain:
                    _x_train.append(im)
                    _y_train.append(target)
                else:
                    _x_test.append(im)
                    _y_test.append(target)

                # imRingHits = np.array(ringHits)
                # imSpotHits = np.array(spotHits)

                # fig.suptitle(r'$\beta_\mathrm{MC} - \beta_\mathrm{Rich} = ' + str(deltaBeta) + r'$')
                # axs[0].cla()
                # axs[0].imshow(im, origin='lower')
                # axs[0].imshow(richOcc, origin='lower', alpha=0.1)
                # axs[0].scatter(ringHits[0], ringHits[1], c='r', alpha=0.2)
                # axs[0].scatter(spotHits[0], spotHits[1], c='b', alpha=0.2)
                # axs[0].scatter(trackHit[0], trackHit[1], c='g', alpha=0.2)
                # axs[1].cla()
                # axs[1].imshow(target, origin='lower')
                # axs[1].imshow(richOcc, origin='lower', alpha=0.1)
                # plt.pause(1)

        return (np.array(_x_train), np.array(_y_train)), (np.array(_x_test), np.array(_y_test))
