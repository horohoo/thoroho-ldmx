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

        c1 = TCanvas()
        c2 = TCanvas()
        c3 = TCanvas()
        c4 = TCanvas()
        
        h1 = TH1F("passtrigger", "passtrigger", 100, 0, 1000)
        h2 = TH1F("passEnergyReq", "passEnergyReq", 100, 0, 1000)
        h3 = TH1F("passTrackerVeto", "passTrackerVeto", 100, 0, 1000)
        h4 = TH1F("passBDT", "passBDT", 100, 0, 1000)
        
        
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

                if Eupstream < 1500:
                    h1.Fill(decayz)

                if Eupstream < 1500 and Edownstream < 2500 and EHcal < 2500 and (Edownstream + EHcal) > 2500: #energy condition for intermediate
                    h2.Fill(decayz)
                        
                    trackerLayerZs = [9.5, 15.5, 24.5, 30.5, 39.5, 45.5, 54.5, 60.5, 93.5, 95.5, 183.5, 185.5]
                    trackerLayersHit = np.zeros(12)
                    trackerHits = np.zeros(6)
                    for hit in RecoilSimHits:
                        for it in SimParticles:
                            if it.first == hit.getTrackID():
                                if it.second.getCharge() != 0:
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
                    for sphit in TargetScoringPlaneHits:
                        if sphit.getPosition()[2] > 0:
                            for it in SimParticles:
                                if it.first == sphit.getTrackID():
                                    if it.second.getPdgID() == 11 and 0 in it.second.getParents():
                                        recoil_E = sphit.getEnergy()


                    if trackinLayer >= 4 and multiTrackinLayer < 4 and recoil_E > 50 and recoil_E < 1200:
                        h3.Fill(decayz)

                        #BDT Variable Stuff Start Here
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
                    
                        #hcal 
                        hits_h = 0
                        xmean_h = 0
                        xstd_h = 0
                        ymean_h = 0
                        ystd_h = 0
                        zmean_h = 0
                        zstd_h = 0
                        isohits_h = 0
                        isoE_h = 0
                        hcal_meangammaproj = 0;
                    
                        for it in SimParticles:
                            for sphit in TargetScoringPlaneHits:
                                if sphit.getPosition()[2] > 0:
                                    if it.first == sphit.getTrackID():
                                        if sphit.getPdgID() == 11 and 0 in it.second.getParents():
                                            x0_gamma = sphit.getPosition()
                                            p_gamma = [-sphit.getMomentum()[0], -sphit.getMomentum()[1], 4000 - sphit.getMomentum()[2]]

                        downstreamrmean_gammaproj = 0
                        downstreamhits_within1 = 0
                        downstreamhits_within2 = 0
                        downstreamhits_within3 = 0
                        downstreamE_within1 = 0
                        downstreamE_within2 = 0
                        downstreamE_within3 = 0
                    
                        #ECAL
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

                        #ECAL standard deviation
                        for hit in EcalRecHits:
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
                    
                        #HCAL
                        for hit in HcalRecHits:
                            hits_h += 1
                            x = hit.getXPos()
                            y = hit.getYPos()
                            z = hit.getZPos()
                            energy = hit.getEnergy()

                            xmean_h += x*energy
                            ymean_h += y*energy
                            zmean_h += z*energy
                        
                            #distance from photon line tracking
                            x_proj = x0_gamma[0] + (z - x0_gamma[2])*p_gamma[0]/p_gamma[2]
                            y_proj = x0_gamma[1] + (z - x0_gamma[2])*p_gamma[1]/p_gamma[2]
                            projdist = math.sqrt((x-x_proj)**2 + (y-y_proj)**2)
                            hcal_meangammaproj += projdist*energy
                        
                            #isolated hits tracking
                            closestpoint = 9999
                            for hit2 in HcalRecHits:
                                if abs(z - hit2.getZPos()) < 1:
                                    sepx = math.sqrt((x-hit2.getXPos())**2)
                                    sepy = math.sqrt((y-hit2.getYPos())**2)
                                    if sepx > 0 and sepx % 50 == 0:
                                        if sepx < closestpoint:
                                            closestpoint = sepx
                                    elif sepy > 0 and sepy % 50 == 0:
                                        if sepy < closestpoint:
                                            closestpoint = sepy
                            if closestpoint > 50:
                                isohits_h += 1
                                isoE_h += energy
                            
                        xmean_h /= EHcal
                        ymean_h /= EHcal
                        zmean_h /= EHcal
                        hcal_meangammaproj /= EHcal
                    
                        #HCAL standard deviation
                        for hit in HcalRecHits:
                            x = hit.getXPos()
                            y = hit.getYPos()
                            z = hit.getZPos()
                            energy = hit.getEnergy()

                            xstd_h += energy*(x-xmean_h)**2
                            ystd_h += energy*(y-ymean_h)**2
                            zstd_h += energy*(z-zmean_h)**2

                        xstd_h = math.sqrt(xstd_h/EHcal)
                        ystd_h = math.sqrt(ystd_h/EHcal)
                        zstd_h = math.sqrt(zstd_h/EHcal)
 

                        evt.append(Etot) #0
                        evt.append(Eupstream) #1
                        evt.append(downstreamhits) #2
                        evt.append(isohits) #3
                        evt.append(isoE)  #4
                        evt.append(xmean) #5
                        evt.append(ymean) #6
                        evt.append(xstd) #7
                        evt.append(ystd) #8
                        evt.append(zstd) #9
                        evt.append(downstreamrmean_gammaproj) #10
                        evt.append(xmean_h) #11
                        evt.append(ymean_h) #12
                        evt.append(xstd_h) #13
                        evt.append(ystd_h) #14
                        evt.append(zstd_h) #15
                        evt.append(hcal_meangammaproj) #16 
                        evt.append(isohits_h) #17
                        evt.append(isoE_h) #18
                       
                        
                        if bdt.predict(xgb.DMatrix(np.vstack((evt,np.zeros_like(evt))),np.zeros(2)))[0] >= 0.99993:
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

                            eventinfo = {'EcalRecHits': ecalrechits,
                                         'HcalRecHits': hcalrechits,
                                         'SimParticles': simparticles,
                                         'TargetScoringPlaneHits': targetscoringplanehits}

                            with open(EDPath + 'eventinfo_{0}.txt'.format(evtcount), 'w') as convert_file:
                                convert_file.write(json.dumps(eventinfo))

                            print("Wrote event {0} to file".format(evtcount))


        """
        h0.Sumw2()
        h1.Sumw2()
        h2.Sumw2()
        h3.Sumw2()
        h4.Sumw2()
        for i in range(h0.GetNbinsX()):
            num1 = h1.GetBinContent(i)
            num2 = h2.GetBinContent(i)
            num3 = h3.GetBinContent(i)
            num4 = h4.GetBinContent(i)
            den = h0.GetBinContent(i)
            if (den != 0):
                h1.SetBinContent(i, num1/den)
                h2.SetBinContent(i, num2/den)
                h3.SetBinContent(i, num3/den)
                h4.SetBinContent(i, num4/den)
                if (num1 != 0):
                    error = num1/den - TEfficiency.ClopperPearson(den, num1, 0.683, False)
                    h1.SetBinError(i, error)
                if (num2 != 0):
                    error = num2/den - TEfficiency.ClopperPearson(den, num2, 0.683, False)
                    h2.SetBinError(i, error)
                if (num3 != 0):
                    error = num3/den - TEfficiency.ClopperPearson(den, num3, 0.683, False)
                    h3.SetBinError(i, error)
                if (num4 != 0):
                    error = num4/den - TEfficiency.ClopperPearson(den, num4, 0.683, False)
                    h4.SetBinError(i, error)
        """
        
        #c0.cd()
        #h0.Draw()
        #c0.Print(EDPath + "totalevents.root")
        #del c0
        del h0
        c1.cd()
        h1.Draw()
        c1.Print(EDPath + "passtrigger.root")
        del c1
        del h1
        c2.cd()
        h2.Draw()
        c2.Print(EDPath + "EnergyReq.root")
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

    parser.add_option('--max_evt', dest='max_evt', type='int', default=450000, help='Max Events to load')
    parser.add_option('--out_dir', dest='out_dir', default='testing', help='Output directory')
    parser.add_option('--bdt_path', dest='bdt_path', default='/sdf/home/h/horoho/ldmx/inter_bdt_weights.pkl', help='BDT model to use')
    parser.add_option('--evtdisplay_path', dest='evtdisplay_path', default='/sfs/qumulo/qhome/wer2ct/LDMX/ldmx-analysis/bdt_plots/signaleffs/5MeV_', help='lol')
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/project/hep_aag/ldmx/ap/visibles/produced/v14/mAp_005', help='name of background file directory')


    (options, args) = parser.parse_args()

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


    print("You set max_evt = ", options.max_evt)

    # load bdt model from pkl file
    gbm = pkl.load(open(options.bdt_path, 'rb'))

    isbkg = True

    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.bkg_dir, options.evtdisplay_path, isbkg, gbm)
