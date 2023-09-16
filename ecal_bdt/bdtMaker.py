import argparse
import importlib
import os
import math
import sys
import random

from LDMX.Framework import EventTree
from LDMX.Framework import ldmxcfg

import matplotlib as plt
import xgboost as xgb
import pickle as pkl
import numpy as np

plt.use('Agg')
from optparse import OptionParser
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##############################################################################

class sampleContainer:
    def __init__(self, dn, maxEvts, trainFrac, isBkg):
        print("Initializing Container!")
        
        self.maxEvts = maxEvts
        self.trainFrac = trainFrac
        self.isBkg = isBkg

        self.events = []
        evtcount = 0
        for filename in os.listdir(dn):
            fn = os.path.join(dn, filename)
            tree = EventTree.EventTree(fn)
            for event in tree:
                evt = []
                Eupstream = 0
                Edownstream = 0
                EHcal = 0
                Etot = 0

                # assign collection names, in case they are different for signal and background
                if isBkg:
                    ecalRecHits = event.EcalRecHits_sim
                    hcalRecHits = event.HcalRecHits_sim
                    SimParticles = event.SimParticles_sim
                    targetScoringPlaneHits = event.TargetScoringPlaneHits_sim
                    
                else:
                    ecalRecHits = event.EcalRecHits_v14
                    hcalRecHits = event.HcalRecHits_v14
                    SimParticles = event.SimParticles_v14
                    targetScoringPlaneHits = event.TargetScoringPlaneHits_v14

                # check that event passes trigger and has activity in the ecal
                for hit in ecalRecHits:
                    Etot += hit.getEnergy()
                    if hit.getZPos() <= 500:
                        Eupstream += hit.getEnergy()
                    else:
                        Edownstream += hit.getEnergy()
                for hit in hcalRecHits:
                    if hit.getZPos() >= 870:
                        EHcal += 12.2*hit.getEnergy()

                if Eupstream < 1500 and Edownstream >= 2500 and EHcal < 2500:
                    # initialize and construct bdt features
                    hits = 0
                    downstreamhits = 0
                    xmean = 0
                    xstd = 0
                    ymean = 0
                    ystd = 0
                    zmean = 0
                    zstd = 0
                    isohits = 0
                    isoE = 0
                    layershit = []
                    for it in SimParticles:
                        for sphit in targetScoringPlaneHits:
                            if sphit.getPosition()[2] > 0:
                                if it.first == sphit.getTrackID():
                                    if isBkg:
                                        if sphit.getPdgID() == 11 and 0 in it.second.getParents():
                                            x0_gamma = sphit.getPosition()
                                            p_gamma = [-sphit.getMomentum()[0], -sphit.getMomentum()[1], 4000 - sphit.getMomentum()[2]]
                                    else:
                                        if sphit.getPdgID() == 622:
                                            x0_gamma = sphit.getPosition()
                                            p_gamma = sphit.getMomentum()
                    downstreamrmean_gammaproj = 0
                    downstreamhits_within1 = 0
                    downstreamhits_within2 = 0
                    downstreamhits_within3 = 0
                    downstreamE_within1 = 0
                    downstreamE_within2 = 0
                    downstreamE_within3 = 0

                    for hit in ecalRecHits:
                        hits += 1
                        x = hit.getXPos()
                        y = hit.getYPos()
                        z = hit.getZPos()
                        r = math.sqrt(x*x + y*y)
                        energy = hit.getEnergy()

                        if not z in layershit:
                            layershit.append(z)

                        xmean += x*energy
                        ymean += y*energy
                        zmean += z*energy

                        if z > 500:
                            downstreamhits += 1
                            Edownstream += energy

                            x_proj = x0_gamma[0] + (z - x0_gamma[2])*p_gamma[0]/p_gamma[2]
                            y_proj = x0_gamma[1] + (z - x0_gamma[2])*p_gamma[1]/p_gamma[2]
                            projdist = math.sqrt((x-x_proj)**2 + (y-y_proj)**2)
                            downstreamrmean_gammaproj += projdist*energy
                            if projdist < 10:
                                downstreamhits_within1 += 1
                                downstreamE_within1 += energy
                            if projdist < 19:
                                downstreamhits_within2 += 1
                                downstreamE_within2 += energy
                            if projdist < 28:
                                downstreamhits_within3 += 1
                                downstreamE_within3 += energy

                        else:
                            Eupstream += energy

                        closestpoint = 9999
                        for hit2 in ecalRecHits:
                            if abs(z - hit2.getZPos()) < 1:
                                isolation = math.sqrt((x-hit2.getXPos())**2 + (y-hit2.getYPos())**2)
                                if isolation > 1 and isolation < closestpoint:
                                    closestpoint = isolation
                        if closestpoint > 9:
                            isohits += 1
                            isoE += energy


                    xmean /= Etot
                    ymean /= Etot
                    zmean /= Etot
                    downstreamrmean_gammaproj /= Edownstream

                    for hit in ecalRecHits:
                        x = hit.getXPos()
                        y = hit.getYPos()
                        z = hit.getZPos()
                        energy = hit.getEnergy()

                        xstd += energy*(x-xmean)**2
                        ystd += energy*(y-ymean)**2
                        zstd += energy*(z-zmean)**2

                    xstd = math.sqrt(xstd/Etot)
                    ystd = math.sqrt(ystd/Etot)
                    zstd = math.sqrt(zstd/Etot)

                    # construct feature vector for event
                    evt.append(Etot) #0
                    evt.append(Eupstream) #1
                    evt.append(Edownstream) #2
                    evt.append(hits) #3
                    evt.append(downstreamhits) #4
                    evt.append(isohits) #5
                    evt.append(isoE) #6
                    evt.append(xmean) #7
                    evt.append(ymean) #8
                    evt.append(xstd) #9
                    evt.append(ystd) #10
                    evt.append(zstd) #11
                    evt.append(downstreamrmean_gammaproj) #12
                    evt.append(downstreamhits_within1) #13
                    evt.append(downstreamhits_within2) #14
                    evt.append(downstreamhits_within3) #15
                    evt.append(downstreamE_within1) #16
                    evt.append(downstreamE_within2) #17
                    evt.append(downstreamE_within3) #18
                    evt.append(len(layershit)) #19


                    self.events.append(evt)
                    evtcount += 1


                if evtcount%50000 == 0:
                    print("Events loaded: ", evtcount)
                if evtcount > self.maxEvts:
                    break
            if evtcount > self.maxEvts:
                break


        print("Initial Event Shape: ", np.shape(self.events))
        new_idx = np.random.permutation(np.arange(np.shape(self.events)[0]))
        self.events = np.array(self.events)
        np.take(self.events, new_idx, axis=0, out=self.events)
        print("Final Event Shape: ", np.shape(self.events))

    # break sample up into training and testing sets
    def constructTrainAndTest(self):
        self.train_x = self.events[0:int(len(self.events)*self.trainFrac)]
        self.test_x = self.events[int(len(self.events)*self.trainFrac):]

        self.train_y = np.zeros(len(self.train_x)) + (self.isBkg == False)
        self.test_y = np.zeros(len(self.test_x)) + (self.isBkg == False)

# merge signal and background sets into one container for training/testing
class mergedContainer:
    def __init__(self, sigContainer, bkgContainer):
        self.train_x = np.vstack((sigContainer.train_x, bkgContainer.train_x))
        self.train_y = np.append(sigContainer.train_y, bkgContainer.train_y)

        self.train_x[np.isnan(self.train_x)] = 0.000
        self.train_y[np.isnan(self.train_y)] = 0.000

        self.test_x = np.vstack((sigContainer.test_x, bkgContainer.test_x))
        self.test_y = np.append(sigContainer.test_y, bkgContainer.test_y)

        self.dtrain = xgb.DMatrix(self.train_x, self.train_y, weight = self.getEventWeights(sigContainer.train_y, bkgContainer.train_y))
        self.dtest = xgb.DMatrix(self.test_x, self.test_y)


    def getEventWeights(self, sig, bkg):
        sigWgt = np.zeros(len(sig)) + 1
        bkgWgt = np.zeros(len(bkg)) + 1. * float(len(sig))/float(len(bkg))
        return np.append(sigWgt, bkgWgt)


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('--seed', dest='seed', type='int', default=2, help='Numpy random seed.')
    parser.add_option('--train_frac', dest='train_frac', default=0.99, help='Fraction of events to use for training')
    parser.add_option('--max_evt', dest='max_evt', type='int', default=450000, help='Max Events to load')
    parser.add_option('--out_name', dest='out_name', default='bdt_v3.1', help='Output Pickle Name')
    parser.add_option('--out_dir', dest='out_dir', default='bdt_v3.1', help='Output directory')
    parser.add_option('--bdt_path', dest='bdt_path', default='/sfs/qumulo/qhome/tgh7hx/ldmx/bdt_v6_weights.pkl', help='BDT model to load in')
    """
    Note on --bdt_path option: this is only used after training a BDT model on
    a combined signal sample for different signal masses. After training, I
    separately test on each mass point by loading in the trained BDT's .pkl
    model. Please contact Tyler (tgh7hx@virginia.edu) for questions about this.
    """
    parser.add_option('--swdir', dest='swdir', default='/sfs/qumulo/qhome/tgh7hx/ldmx/ldmx-sw/install', help='ldmx-sw build directory')
    parser.add_option('--eta', dest='eta', type='float', default=0.023, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number', type='int', default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth', type='int', default=20, help='Max Tree Depth')
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/scratch/tgh7hx/v3.2.0_ecalPN_tskim-batch1/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/scratch/tgh7hx/signal_train/', help='name of signal file directory')


    (options, args) = parser.parse_args()

    np.random.seed(options.seed)

    adds = 0
    Check = True
    while Check:
        if not os.path.exists(options.out_dir+'_'+str(adds)):
            try:
                os.makedirs(options.out_dir+'_'+str(adds))
                Check = False
            except:
                Check = True
        else:
            adds += 1


    print("Random seed is ", options.seed)
    print("You set max_ect = ", options.max_evt)
    print("You set train frac = ", options.train_frac)
    print("You set tree number = ", options.tree_number)
    print("You set max tree depth = ", options.depth)
    print("You set eta = ", options.eta)

    # load in signal events
    print('Loading sig_file = ', options.sig_dir)
    sigContainer = sampleContainer(options.sig_dir, options.max_evt, options.train_frac, False)
    sigContainer.constructTrainAndTest()

    # load in background events
    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.bkg_dir, options.max_evt, options.train_frac, True)
    bkgContainer.constructTrainAndTest()

    eventContainer = mergedContainer(sigContainer, bkgContainer)

    # bdt parameters
    params = {"objective": "binary:logistic",
              "eta": options.eta,
              "max_depth": options.depth,
              "min_child_weight": 20,
              "silent": 1,
              "subsample": 0.9,
              "colsample_bytree": 0.85,
              "eval_metric": 'auc',
              "seed": 1,
              "nthread": 1,
              "verbosity": 1,
              "early_stopping_rounds": 10}


    num_trees = options.tree_number
    evallist = [(eventContainer.dtest,'eval'), (eventContainer.dtrain,'train')]
    gbm = xgb.train(params, eventContainer.dtrain, num_trees, evallist) # use this option for training a new bdt
    #gbm = pkl.load(open(options.bdt_path, 'rb')) # use this option for testing on a previously trained bdt

    preds = gbm.predict(eventContainer.dtest)
    fpr, tpr, threshold = metrics.roc_curve(eventContainer.test_y, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print("Final validation AUC = ", roc_auc)

    preds_train = gbm.predict(eventContainer.dtrain)
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(eventContainer.train_y, preds_train)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    print("Final validation AUC (train data) = ", roc_auc_train)


    np.savetxt(options.out_dir+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_validation_preds.txt', np.c_[preds, eventContainer.test_y])
    np.savetxt(options.out_dir+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_validation_threetuples.txt', np.c_[fpr, tpr, threshold])
    np.savetxt(options.out_dir+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_train_threetuples.txt', np.c_[fpr_train, tpr_train, threshold_train])
    output = open(options.out_dir+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_weights'+'.pkl', 'wb')
    pkl.dump(gbm, output)

    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_dir+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_fimportance.png', dpi=500, bbox_inches='tight', pad_inches=0.5)

    print("Files saved in: ", options.out_dir+'_'+str(adds))
