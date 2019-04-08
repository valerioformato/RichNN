import numpy as np
import uproot as up
import matplotlib.pyplot as plt

rootfile = up.open('/Volumes/AMS_Disk/DBar/test/test_ML/RichNN/data/training_data.root')
data_tree = rootfile['VFRichNNBuildTree/TrainingData']

dim = 145
pad = 20

imRichOcc = np.zeros((dim + 2*pad, dim + 2*pad))

for arrays in data_tree.iterate():
    for iEv in range(0, len(arrays['RichHits'])):
        nHits = arrays['RichHits'][iEv]

        for ihit in range(0, nHits):
            xPix = arrays['RichHits.xPix'][iEv][ihit] + pad
            yPix = arrays['RichHits.yPix'][iEv][ihit] + pad

            imRichOcc[yPix][xPix] += 1

imRichOcc = np.clip(imRichOcc, a_min=0, a_max=1)

import scipy.misc
scipy.misc.toimage(imRichOcc, cmin=0.0, cmax=1.0).save('RichOcc.jpg')
