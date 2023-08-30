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

                if isBkg:
                    EcalRecHits = event.EcalRecHits_sim
                    HcalRecHits = event.HcalRecHits_sim
                    SimParticles = event.SimParticles_sim
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_sim
                    
                else:
                    EcalRecHits = event.EcalRecHits_v14
                    HcalRecHits = event.HcalRecHits_v14
                    SimParticles = event.SimParticles_v14
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_v14

                # Find energy upstream of z=500mm and check if trigger condition is met
                Eupstream = 0
                Edownstream = 0
                E_hcal = 0

                for hit in HcalRecHits:
                    if hit.getZPos() >= 870:
                        E_hcal += 12*hit.getEnergy()
                        
                for hit in EcalRecHits:
                    if hit.getZPos() > 500:
                        Edownstream += hit.getEnergy()
                    else:
                        Eupstream += hit.getEnergy()
                                
                                
                if Eupstream < 1500 and E_hcal >= 2500 and Edownstream < 2500 :

                    hits = 0
                    downstreamhits = 0
                    widedownstreamhits = 0
                    rsqarr = []
                    zsqarr=[]
                    xarr = []
                    yarr = []
                    zarr = []
                    isohits = 0
                    isoE = 0
                    Eupstream = 0
                    xmean=0
                    ymean=0
                    zmean=0
                    xstd=0
                    ystd=0
                    zstd=0
                    
                  
                    Etot = 0
                    centralE=0
                    layershit = []
                    for it in SimParticles:
                        for sphit in targetScoringPlaneHits:
                            if sphit.getPosition()[2] > 0:
                                if it.first == sphit.getTrackID():
                                    if sphit.getPdgID() == 11 and 0 in it.second.getParents():
                                        x0_gamma = sphit.getPosition()
                                        p_gamma = [-sphit.getMomentum()[0], -sphit.getMomentum()[1], 4000 - sphit.getMomentum()[2]]
                    downstreamrmean_gammaproj = 0 
                    
                    
                    for hit in HcalRecHits:
                        hits += 1
                        x = hit.getXPos()
                        y = hit.getYPos()
                        z = hit.getZPos()
                        r = math.sqrt(x*x + y*y)
                        energy = hit.getEnergy()
                        xarr.append(x)
                        yarr.append(y)
                        zarr.append(z)
                        Etot += energy
                        rsqarr.append(r*r)
                        zsqarr.append(z*z)
                        
                        
                        xmean += x*energy
                        ymean += y*energy
                        zmean += z*energy

                        x_proj = x0_gamma[0] + (z - x0_gamma[2])*p_gamma[0]/p_gamma[2]
                        y_proj = x0_gamma[1] + (z - x0_gamma[2])*p_gamma[1]/p_gamma[2]
                        projdist = math.sqrt((x-x_proj)**2 + (y-y_proj)**2)
                        downstreamrmean_gammaproj += projdist*energy
                        
                        closestpoint = 9999
                        for hit2 in event.HcalRecHits_v12:
                            if abs(z - hit2.getZPos()) < 1:
                                sepx = math.sqrt((x-hit2.getXPos())**2)
                                sepy = math.sqrt((y-hit2.getYPos())**2)
                                if sepx > 0 and sepx%50 == 0:
                                    if sepx < closestpoint:
                                        closestpoint = sepx
                                elif sepy > 0 and sepy%50 == 0:
                                    if sepy < closestpoint:
                                        closestpoint = sepy
                        if closestpoint > 50:
                            isohits += 1
                            isoE += energy
                            
                    xmean /= Etot
                    ymean /= Etot
                    zmean /= Etot
                        
                    downstreamrmean_gammaproj /= Etot    
                     
                    for hit in hcalRecHits:
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
    


                    # Fill event with features to train BDT
                    evt.append(math.sqrt(np.mean(rsqarr))) #0
                    evt.append(math.sqrt(np.mean(zsqarr))) #1
                  
                    evt.append(xstd) #2
                    evt.append(ystd) #3
                    evt.append(zstd) #4
                               
                    evt.append(np.std(zarr)) #5
                    evt.append(np.std(xarr)) #6
                    evt.append(np.std(yarr)) #7
                    
                    evt.append(isohits) #8
                    evt.append(isoE) #9
                    evt.append(hits) #10
                    
                    evt.append(Etot) #11
                    
                    evt.append(downstreamrmean_gammaproj) #12

                    self.events.append(evt)
                    evtcount += 1


                if evtcount > self.maxEvts:
                    break
            if evtcount > self.maxEvts:
                break


        print("Initial Event Shape: ", np.shape(self.events))
        new_idx = np.random.permutation(np.arange(np.shape(self.events)[0]))
        self.events = np.array(self.events)
        np.take(self.events, new_idx, axis=0, out=self.events)
        print("Final Event Shape: ", np.shape(self.events))

    def constructTrainAndTest(self):
        
        
        
        self.train_x = self.events[0:int(len(self.events)*self.trainFrac)]
        self.test_x = self.events[int(len(self.events)*self.trainFrac):]

        self.train_y = np.zeros(len(self.train_x)) + (self.isBkg == False)
        self.test_y = np.zeros(len(self.test_x)) + (self.isBkg == False)

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

    parser.add_option('--seed', dest='seed', type='int', default=4, help='Numpy random seed.')
    parser.add_option('--train_frac', dest='train_frac', default=0.80, help='Fraction of events to use for training')
    parser.add_option('--max_evt', dest='max_evt', type='int', default=500000, help='Max Events to load')
    parser.add_option('--out_name', dest='out_name', default='bdt', help='Output Pickle Name')
    
    
    parser.add_option('--swdir', dest='swdir', default='/sfs/qumulo/qhome/lbc2qnt/ldmx-sw/install', help='ldmx-sw build directory')
    parser.add_option('--eta', dest='eta', type='float', default=0.023, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number', type='int', default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth', type='int', default=6, help='Max Tree Depth')
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/project/hep_aag/ldmx/background/PN_Ecal/sim/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/project/hep_aag/ldmx/ap/visibles/produced/mAp_050/', help='name of signal file directory')


    (options, args) = parser.parse_args()

    np.random.seed(options.seed)

    adds = 0
    Check = True
    while Check:
        if not os.path.exists(options.out_name+'_'+str(adds)):
            try:
                os.makedirs(options.out_name+'_'+str(adds))
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


    print('Loading sig_file = ', options.sig_dir)
    sigContainer = sampleContainer(options.sig_dir, options.max_evt, options.train_frac, False)
    sigContainer.constructTrainAndTest()

    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.bkg_dir, options.max_evt, options.train_frac, True)
    bkgContainer.constructTrainAndTest()

    eventContainer = mergedContainer(sigContainer, bkgContainer)

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
    gbm = xgb.train(params, eventContainer.dtrain, num_trees, evallist)


    preds = gbm.predict(eventContainer.dtest)
    fpr, tpr, threshold = metrics.roc_curve(eventContainer.test_y, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    
    preds_train = gbm.predict(eventContainer.dtrain)
    fpr_t, tpr_t, threshold_t = metrics.roc_curve(eventContainer.train_y, preds_train)
    np.savetxt(options.out_name+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_training_threetuples.txt', np.c_[fpr_t, tpr_t, threshold_t])
        
    
    
    print("Final validation AUC = ", roc_auc)
    np.savetxt(options.out_name+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_validation_preds.txt', np.c_[preds, eventContainer.test_y])
    np.savetxt(options.out_name+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_validation_threetuples.txt', np.c_[fpr, tpr, threshold])
    output = open(options.out_name+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_weights'+'.pkl', 'wb')
    pkl.dump(gbm, output)

    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_name+'_'+str(adds)+'/'+options.out_name+'_'+str(adds)+'_fimportance.png', dpi=500, bbox_inches='tight', pad_inches=0.5)

    print("Files saved in: ", options.out_name+'_trial3'+str(adds))
