import numpy as np
import uproot as up
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)

rootfile = up.open('/Volumes/AMS_Disk/DBar/test/test_ML/RichNN/data/training_data.root')
data_tree = rootfile['VFRichNNBuildTree/TrainingData']

dim = 145
pad = 20

richOcc = plt.imread('RichOcc.jpg')

fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10,5.5))
plt.ion()
# plt.show()

for arrays in data_tree.iterate():
    for iEv in range(0, len(arrays['RichHits'])):
        im = np.zeros((dim + 2*pad, dim + 2*pad, 1))
        target = np.zeros((dim + 2*pad, dim + 2*pad, 2)) # for multiclass the last dim will be >1
                                                         # for now one layer is hit position, the other is ring center
        nHits = arrays['RichHits'][iEv]
        # print nHits

        ringHits = [[], []]
        spotHits = [[], []]
        trackHit = [[], []]

        xTrPix = arrays['TrRichPMTIntXPix'][iEv] + pad
        yTrPix = arrays['TrRichPMTIntYPix'][iEv] + pad

        deltaBeta = arrays['MCBeta'][iEv] - arrays['BetaRich'][iEv]

        for ihit in range(0, nHits):
            xPix = arrays['RichHits.xPix'][iEv][ihit] + pad
            yPix = arrays['RichHits.yPix'][iEv][ihit] + pad

            # print ihit, xPix, yPix, arrays['RichHits.nPhElUncorr'][iEv][ihit]
            im[yPix][xPix] = [arrays['RichHits.nPhElUncorr'][iEv][ihit]]

            if (arrays['RichHits.status'][iEv][ihit] & 1) > 0:
                ringHits[0].append(xPix)
                ringHits[1].append(yPix)
                target[yPix][xPix] = [1, 0]

            if (arrays['RichHits.status'][iEv][ihit] & 1<<30) > 0:
                spotHits[0].append(xPix)
                spotHits[1].append(yPix)

            trackHit[0].append(xTrPix)
            trackHit[1].append(yTrPix)
            target[yTrPix][xTrPix] = [0, 1]

        imRingHits = np.array(ringHits)
        imSpotHits = np.array(spotHits)

        fig.suptitle(r'$\beta_\mathrm{MC} - \beta_\mathrm{Rich} = ' + str(deltaBeta) + r'$')
        axs[0].cla()
        axs[0].imshow(im[:,:,0], origin='lower')
        axs[0].imshow(richOcc, origin='lower', alpha=0.1)
        axs[0].scatter(ringHits[0], ringHits[1], c='r', alpha=0.2)
        axs[0].scatter(spotHits[0], spotHits[1], c='b', alpha=0.2)
        axs[0].scatter(trackHit[0], trackHit[1], c='g', alpha=0.2)
        axs[1].cla()
        axs[1].imshow(target[:,:,0], origin='lower')
        axs[1].imshow(richOcc, origin='lower', alpha=0.1)
        plt.pause(5)
