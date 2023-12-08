import argparse
import importlib
import os
import math
import statistics
import sys
import random
import json

from LDMX.Framework import EventTree
from LDMX.Framework import ldmxcfg

from ROOT import TCanvas, TH3F, TH2F, TH1F, TEfficiency, TGraph2D

import matplotlib as plt
import xgboost as xgb
import pickle as pkl
import numpy as np
import pandas as pd

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
        h2 = TH1F("ecalEnergyReq", "ecalEnergyReq", 200, 0, 2000)
        h3 = TH1F("passTrackerVeto", "passTrackerVeto", 200, 0, 2000)
        h4 = TH1F("passBDT", "passBDT", 200, 0, 2000)
        
        for filename in os.listdir(dn):
            fn = os.path.join(dn, filename)
            tree = EventTree.EventTree(fn)
            for event in tree:
                #evtcount += 1

                if isBkg:
                    EcalRecHits = event.EcalRecHits_sim
                    HcalRecHits = event.HcalRecHits_sim
                    SimParticles = event.SimParticles_sim
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_sim
                    RecoilSimHits = event.RecoilSimHits_sim

                else:
                    EcalRecHits = event.EcalRecHits_v14
                    HcalRecHits = event.HcalRecHits_v14
                    SimParticles = event.SimParticles_v14
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_v14
                    RecoilSimHits = event.RecoilSimHits_v14

                evt = []

                Eupstream = 0
                Edownstream = 0
                Etot = 0
                EHcal = 0

                decayz = 1
                for it in SimParticles:
                    parents = it.second.getParents()
                    for track_id in parents:
                        if track_id == 0 and it.second.getPdgID() == -11:
                            decayz = it.second.getVertex()[2]

                for hit in EcalRecHits:
                    Etot += hit.getEnergy()
                    if hit.getZPos() < 500:
                        Eupstream += hit.getEnergy()
                    else:
                        Edownstream += hit.getEnergy()

                for hit in HcalRecHits:
                    if hit.getZPos() >= 870:
                        EHcal += 12.2*hit.getEnergy()

                h0.Fill(decayz)
                if Eupstream < 1500:
                    h1.Fill(decayz)

                if Eupstream < 1500 and Edownstream >= 2500 and EHcal < 2500:
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

                    #if trackinLayer >= 4 and multiTrackinLayer < 4 and recoil_E > 50 and recoil_E < 1200:
                    #    h3.Fill(decayz)

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
                    for sphit in TargetScoringPlaneHits:
                        if sphit.getPosition()[2] > 0:
                            for it in SimParticles:
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

                    for hit in EcalRecHits:
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
                        for hit2 in EcalRecHits:
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

                    for hit in EcalRecHits:
                        xstd += energy*(xmean-hit.getXPos())**2
                        ystd += energy*(ymean-hit.getYPos())**2
                        zstd += energy*(zmean-hit.getZPos())**2


                    evt.append(Etot)
                    evt.append(Eupstream)
                    evt.append(Edownstream)
                    evt.append(hits)
                    evt.append(downstreamhits)
                    evt.append(isohits)
                    evt.append(isoE)
                    evt.append(xmean)
                    evt.append(ymean)
                    evt.append(xstd)
                    evt.append(ystd)
                    evt.append(zstd)
                    evt.append(downstreamrmean_gammaproj)
                    evt.append(downstreamhits_within1)
                    evt.append(downstreamhits_within2)
                    evt.append(downstreamhits_within3)
                    evt.append(downstreamE_within1)
                    evt.append(downstreamE_within2)
                    evt.append(downstreamE_within3)
                    evt.append(len(layershit))

                        
                    if bdt.predict(xgb.DMatrix(np.vstack((evt,np.zeros_like(evt))),np.zeros(2)))[0] >= 0.999971:
                        h4.Fill(decayz)
                        evtcount += 1
                        if trackinLayer >= 4 and multiTrackinLayer < 4 and recoil_E > 50 and recoil_E < 1200:
                            h3.Fill(decayz)

                        
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


                        recoilsimhits = []
                        for hit in RecoilSimHits:
                            recoilsimhits.append([[hit.getPosition()[0], hit.getPosition()[1], hit.getPosition()[2]],
                                                  [hit.getMomentum()[0], hit.getMomentum()[1], hit.getMomentum()[2]]])

                        eventinfo = {'EcalRecHits': ecalrechits,
                                     'HcalRecHits': hcalrechits,
                                     'SimParticles': simparticles,
                                     'TargetScoringPlaneHits': targetscoringplanehits,
                                     'RecoilSimHits': recoilsimhits}

                        
                        with open(EDPath + 'eventinfo_{0}.txt'.format(evtcount), 'w') as convert_file:
                            convert_file.write(json.dumps(eventinfo))
                            
                        print("Wrote event {0} to file".format(evtcount))
                        
                            
        
        c0.cd()
        h0.Draw()
        c0.Print(EDPath + "totalevents.root")
        del c0
        del h0
        c1.cd()
        h1.Draw()
        c1.Print(EDPath + "passtrigger.root")
        del c1
        del h1
        c2.cd()
        h2.Draw()
        c2.Print(EDPath + "EcalEnergyReq.root")
        del c2
        del h2
        c3.cd()
        h3.Draw()
        c3.Print(EDPath + "passtrackerveto.root")
        del c3
        del h3
        c4.cd()
        h4.Draw()
        c4.Print(EDPath + "passBDT.root")
        del c4
        del h4
        

if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('--bdt_path', dest='bdt_path', default='/sdf/home/h/horoho/ldmx/thoroho-ldmx/ecal_bdt/cindy.pkl', help='BDT model to use')
    parser.add_option('--evtdisplay_path', dest='evtdisplay_path', default='/sfs/qumulo/qhome/tgh7hx/ldmx/signaleffs/ecal_005_', help='Where to put events that pass veto')
    parser.add_option('--swdir', dest='swdir', default='/sdf/home/h/horoho/ldmx/ldmx-sw/install', help='ldmx-sw build directory')
    
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/scratch/tgh7hx/ldmx/batch/targetPN/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/sdf/group/ldmx/data/ap_visibles/v14/mAp_005/', help='name of signal file directory')


    (options, args) = parser.parse_args()


    # load bdt model from pkl file
    gbm = pkl.load(open(options.bdt_path, 'rb'))

    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.bkg_dir, options.evtdisplay_path, True, gbm)
