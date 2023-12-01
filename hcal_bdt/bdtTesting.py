import argparse
import importlib
import os
import math
import sys
import random
import json

from LDMX.Framework import EventTree
from LDMX.Framework import ldmxcfg

from ROOT import TCanvas, TH3F, TH1F, TEfficiency, TGraph2D, TTree, TFile

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
    def __init__(self, dn, EDPath, isBkg, bdt):
        print("Initializing Container!")
        self.isBkg = isBkg
        self.events = []
        evtcount = 0

        c0 = TCanvas()
        c1 = TCanvas()
        c2 = TCanvas()
        c3 = TCanvas()
        c4 = TCanvas()

        h0 = TH1F("totalevents", "totalevents", 200, 0, 2000)
        h1 = TH1F("passtrigger", "passtrigger", 200, 0, 2000)
        h2 = TH1F("hcalEnergyReq", "hcalEnergyReq", 200, 0, 2000)
        h3 = TH1F("passTrackerVeto", "passTrackerVeto", 200, 0, 2000)
        h4 = TH1F("passBDT", "passBDT", 200, 0, 2000)

        for filename in os.listdir(dn):
            fn = os.path.join(dn, filename)
            tree = EventTree.EventTree(fn)
            for event in tree:

                if isBkg:
                    EcalRecHits = event.EcalRecHits_sim
                    HcalRecHits = event.HcalRecHits_sim
                    SimParticles = event.SimParticles_sim
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_sim
                    RecoilSimHits = event.RecoilSimHits_sim
                    EcalVeto = event.EcalVeto_sim

                else:
                    EcalRecHits = event.EcalRecHits_v14
                    HcalRecHits = event.HcalRecHits_v14
                    SimParticles = event.SimParticles_v14
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_v14
                    RecoilSimHits = event.RecoilSimHits_v14
                    EcalVeto = event.EcalVeto_v14
                    
 
                evt = []

                Eupstream = 0
                Edownstream = 0
                EHcal = 0

                decayz = 1
                for it in SimParticles:
                    parents = it.second.getParents()
                    for track_id in parents:
                        if track_id == 0 and it.second.getPdgID() == -11:
                            decayz = it.second.getVertex()[2]

                for hit in HcalRecHits:
                    if hit.getZPos() >= 800:
                        EHcal += 12*hit.getEnergy()
                        
                for hit in EcalRecHits:
                    if hit.getZPos() > 500:
                        Edownstream += hit.getEnergy()
                    else:
                        Eupstream += hit.getEnergy()

                h0.Fill(decayz)
                if Eupstream < 1500:
                    h1.Fill(decayz)


                if Eupstream < 1500 and Edownstream < 2500 and EHcal >= 2500:
                    h2.Fill(decayz)

                    trackerLayerZs = [9.5, 15.5, 24.5, 30.5, 39.5, 45.5, 54.5, 60.5, 93.5, 95.5, 183.5, 185.5]
                    trackerLayersHit = np.zeros(12)
                    trackerHits = np.zeros(6)
                    for hit in RecoilSimHits:
                        p = math.sqrt((hit.getMomentum()[0])**2 + (hit.getMomentum()[1])**2 + (hit.getMomentum()[2])**2)
                        if p >= 50:
                            for i in range(12):
                                if abs(hit.getPosition()[2] - trackerLayerZs[i]) < 1:
                                    trackerLayersHit[i] += 1
                    for i in range(6):
                        hit1count = trackerLayersHit[2*i]
                        hit2count = trackerLayersHit[2*i+1]
                        if hit1count >= hit2count:
                            trackerHits[i] = hit1count
                        else:
                            trackerHits[i] = hit2count
                    trackinLayer = 0
                    multiTrackinLayer = 0
                    for i in range(6):
                        if trackerHits[i] > 1:
                            multiTrackinLayer += 1
                        if trackerHits[i] > 0:
                            trackinLayer += 1
                    recoil_E = 0
                    e_list = []
                    for sphit in TargetScoringPlaneHits:
                        if sphit.getPosition()[2] > 0:
                            for it in SimParticles:
                                if it.first == sphit.getTrackID():
                                    if it.second.getPdgID() == 11:
                                        e_list.append(sphit.getEnergy())

                    if len(e_list) > 0:
                        recoil_E = max(e_list)

                    if trackinLayer >= 4 and multiTrackinLayer < 4 and recoil_E > 50 and recoil_E < 1200:
                        h3.Fill(decayz)
                        
                    hits = 0
                    isohits = 0
                    isoE = 0

                    xmean = 0
                    ymean = 0
                    zmean = 0
                    rmean = 0
                    
                    xmean_equal = 0
                    ymean_equal = 0
                    zmean_equal = 0
                    rmean_equal = 0

                    rms_r = 0
                    rms_z = 0
                    
                    xstd = 0
                    ystd = 0
                    zstd = 0

                    xstd_equal = 0
                    ystd_equal = 0
                    zstd_equal = 0
                        
                    Etot = 0
                    
                    for it in SimParticles:
                        for sphit in TargetScoringPlaneHits:
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
                        
                        
                    for hit in HcalRecHits:
                        hits += 1
                        x = hit.getXPos()
                        y = hit.getYPos()
                        z = hit.getZPos()
                        r = math.sqrt(x*x + y*y)
                        
                        energy = hit.getEnergy()
                        Etot += energy

                        xmean += x*energy
                        ymean += y*energy
                        zmean += z*energy

                        xmean_equal += x
                        ymean_equal += y
                        zmean_equal += z

                        rms_r += r*r
                        rms_z += z*z
                                
                        x_proj = x0_gamma[0] + (z - x0_gamma[2])*p_gamma[0]/p_gamma[2]
                        y_proj = x0_gamma[1] + (z - x0_gamma[2])*p_gamma[1]/p_gamma[2]
                        projdist = math.sqrt((x-x_proj)**2 + (y-y_proj)**2)
                        downstreamrmean_gammaproj += projdist*energy
                        
                        closestpoint = 9999
                        for hit2 in HcalRecHits:
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
                    rmean /= Etot

                    xmean_equal /= hits
                    ymean_equal /= hits
                    zmean_equal /= hits
                    rmean_equal /= hits
                    rms_r /= hits
                    rms_z /= hits
                        
                    downstreamrmean_gammaproj /= Etot    
                     
                    for hit in HcalRecHits:
                        x = hit.getXPos()
                        y = hit.getYPos()
                        z = hit.getZPos()
                        energy = hit.getEnergy()

                        xstd += energy*(x-xmean)**2
                        ystd += energy*(y-ymean)**2
                        zstd += energy*(z-zmean)**2

                        xstd_equal += (x-xmean_equal)**2
                        ystd_equal += (y-ymean_equal)**2
                        zstd_equal += (z-zmean_equal)**2

                    xstd = math.sqrt(xstd/Etot)
                    ystd = math.sqrt(ystd/Etot)
                    zstd = math.sqrt(zstd/Etot)

                    xstd_equal = math.sqrt(xstd_equal/hits)
                    ystd_equal = math.sqrt(ystd_equal/hits)
                    zstd_equal = math.sqrt(zstd_equal/hits)

                    rms_r = math.sqrt(rms_r)
                    rms_z = math.sqrt(rms_z)
                            

                    evt.append(rms_r) #0    
                    evt.append(rms_z) #1
        
                    evt.append(xstd) #2
                    evt.append(ystd) #3
                    evt.append(zstd) #4

                    evt.append(xmean) #5
                    evt.append(ymean) #6
                    evt.append(rmean) #7
                        
                    evt.append(zstd_equal) #8   
                    evt.append(xstd_equal) #9
                    evt.append(ystd_equal) #10

                    evt.append(rmean_equal) #11
                    evt.append(xmean_equal) #12
                    evt.append(ymean_equal) #13
                    
                    evt.append(isohits) #14
                    evt.append(isoE) #15
                    evt.append(hits) #16
                    evt.append(Etot) #17
                    evt.append(downstreamrmean_gammaproj) #18
                            

                    if bdt.predict(xgb.DMatrix(np.vstack((evt,np.zeros_like(evt))),np.zeros(2)))[0] > 0.9999525:
                        h4.Fill(decayz)
                        evtcount += 1

                        ecalrechits = []
                        for hit in EcalRecHits:
                            ecalrechits.append([[hit.getXPos(), hit.getYPos(), hit.getZPos()],
                                                hit.getEnergy()])

                        hcalrechits = []
                        for hit in HcalRecHits:
                            hcalrechits.append([[hit.getXPos(), hit.getYPos(), hit.getZPos()],
                                                hit.getEnergy()])
                            
                        simparticles = []
                        for it in SimParticles:
                            daughters = []
                            for daughter in it.second.getDaughters():
                                daughters.append(daughter)
                            parents = []
                            for parent in it.second.getParents():
                                parents.append(parent)

                            simparticles.append([it.first,
                                                 it.second.getEnergy(),
                                                 it.second.getPdgID(),
                                                 [it.second.getVertex()[0], it.second.getVertex()[1], it.second.getVertex()[2]],
                                                 [it.second.getEndPoint()[0], it.second.getEndPoint()[1], it.second.getEndPoint()[2]],
                                                 [it.second.getMomentum()[0], it.second.getMomentum()[1], it.second.getMomentum()[2]],
                                                 it.second.getMass(),
                                                 it.second.getCharge(),
                                                 daughters,
                                                 parents])

                        targetscoringplanehits = []
                        for sphit in TargetScoringPlaneHits:
                            targetscoringplanehits.append([[sphit.getPosition()[0], sphit.getPosition()[1], sphit.getPosition()[2]],
                                                           sphit.getEnergy(),
                                                           [sphit.getMomentum()[0], sphit.getMomentum()[1], sphit.getMomentum()[2]],
                                                           sphit.getTrackID(),
                                                           sphit.getPdgID()])

                        ecalveto = []
                        ecalveto.append(EcalVeto.getDisc(),
                                        EcalVeto.getNStraightTracks(),
                                        EcalVeto.getNLinRegTracks()]


                        eventinfo = {'EcalRecHits': ecalrechits,
                                     'HcalRecHits': hcalrechits,
                                     'SimParticles': simparticles,
                                     'TargetScoringPlaneHits': targetscoringplanehits,
                                     'EcalVeto': ecalveto}
                        """
                        with open(EDPath + 'eventinfo_{0}.txt'.format(evtcount), 'w') as convert_file:
                            convert_file.write(json.dumps(eventinfo))
                            
                        print("Wrote event {0} to file".format(evtcount))
                        """        

        c1.cd()
        h1.Draw()
        c1.Print(EDPath + "passtrigger.root")
        c2.cd()
        h2.Draw()
        c2.Print(EDPath + "HcalEnergyReq.root")
        c3.cd()
        h3.Draw()
        c3.Print(EDPath + "passTrackerVeto.root")
        c4.cd()
        h4.Draw()
        c4.Print(EDPath + "passBDT.root")

        del c1
        del c2
        del c3
        del c4
        del h1
        del h2
        del h3
        del h4



if __name__ == '__main__':

    parser = OptionParser()
    
    parser.add_option('--bdt_path', dest='bdt_path', default='/sfs/qumulo/qhome/tgh7hx/ldmx/hcal_bdt_weights.pkl', help='BDT model to use')
    parser.add_option('--evtdisplay_path', dest='evtdisplay_path', default='/sfs/qumulo/qhome/tgh7hx/ldmx/hcal_bdt/', help='Where to put events that pass veto')
    
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/sdf/group/ldmx/data/mc23/v14/4.0GeV/v3.2.0_ecalPN_tskim-batch3/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/scratch/tgh7hx/mAp_005_test/', help='name of signal file directory')


    (options, args) = parser.parse_args()

    # load bdt model from pkl file
    gbm = pkl.load(open(options.bdt_path, 'rb'))

    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.sig_dir, options.evtdisplay_path, False, gbm)
