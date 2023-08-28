#New energy contraint, redo bdtmaker with constraints and variables,redo isohits and add z density etc

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
        c1 = TCanvas()
        c2 = TCanvas()
        c3 = TCanvas()
        c4 = TCanvas()
        h1 = TH1F("passtrigger", "passtrigger", 3, 0, 2)
        h2 = TH1F("hcalEnergyReq", "hcalEnergyReq", 3, 0, 2)
        h3 = TH1F("passTrackerVeto", "passTrackerVeto", 3, 0, 2)
        h4 = TH1F("passBDT", "passBDT", 3, 0, 2)

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
                EHcal = 0

                for hit in HcalRecHits:
                    if hit.getZPos() >= 800:
                        EHcal += 12*hit.getEnergy()
                        
                for hit in EcalRecHits:
                    if hit.getZPos() > 500:
                        Edownstream += hit.getEnergy()
                    else:
                        Eupstream += hit.getEnergy()

                if Eupstream < 1500:
                    h1.Fill(1)


                if Eupstream < 1500 and Edownstream < 2500 and EHcal >= 2500:
                    h2.Fill(1)

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
                        h3.Fill(1)
                        
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
                        for it in event.SimParticles_sim:
                            for sphit in event.TargetScoringPlaneHits_sim:
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
                        
                        downstreamrmean_gammaproj /= Etot    
                     
                        for hit in HcalRecHits:
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
                            

                        evt.append(math.sqrt(np.mean(rsqarr)))    
                        evt.append(math.sqrt(np.mean(zsqarr)))
        
                        evt.append(xstd) 
                        evt.append(ystd)
                        evt.append(zstd)
                        
                        evt.append(np.std(zarr))    
                        evt.append(np.std(xarr))
                        evt.append(np.std(yarr))
                    
                        evt.append(isohits)
                        evt.append(isoE)
                        evt.append(hits)
                        evt.append(Etot)
                        evt.append(downstreamrmean_gammaproj)
                            

                        if bdt.predict(xgb.DMatrix(np.vstack((evt,np.zeros_like(evt))),np.zeros(2)))[0] > 0.99998:
                            h4.Fill(1)
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
                                c3d = TCanvas()
                                h3_vol = TH3F(str(evtcount), str(evtcount), 2, -1500, 1500, 2, -1500, 1500, 2, 0, 6000)
                                h3_hcal = TH3F("hcalhits", "hcalhits", 100, -1500, 1500, 100, -1500, 1500, 120, 0, 6000)
                                
                                for hit in event.HcalRecHits_sim:
                                    h3_hcal.Fill(hit.getXPos(), hit.getYPos(), hit.getZPos(), 12.2*hit.getEnergy())
                                    c3d.cd()
                                    h3_vol.Draw("BOX")
                                    h3_hcal.Draw("BOX2 Z")

                                photondaughters = []
                                for it in event.SimParticles_sim:
                                    for sphit in event.TargetScoringPlaneHits_sim:
                                        if sphit.getPosition()[2] > 0:
                                            if it.first == sphit.getTrackID():
                                                if it.second.getPdgID() == 22:
                                                    if it.second.getEnergy() >= 2500:
                                                        photondaughters.append(np.array(it.second.getDaughters()))

                                                if it.second.getPdgID() == 11 and 0 in it.second.getParents():
                                                    tg_recoilproj = TGraph2D()
                                                    tg_recoilproj.SetPoint(0,
                                                                           sphit.getPosition()[0],
                                                                           sphit.getPosition()[1],
                                                                           sphit.getPosition()[2])
                                                    tg_recoilproj.SetPoint(1,
                                                                           sphit.getPosition()[0] + (750 - sphit.getPosition()[2])*sphit.getMomentum()[0]/sphit.getMomentum()[2],
                                                                           sphit.getPosition()[1] + (750 - sphit.getPosition()[2])*sphit.getMomentum()[1]/sphit.getMomentum()[2],
                                                                           750)
                                                    tg_recoilproj.SetLineColor(4)
                                                    tg_recoilproj.SetLineWidth(3)
                                                    tg_recoilproj.Draw("LINE same")

                                                    tg_bremproj = TGraph2D()
                                                    tg_bremproj.SetPoint(0,
                                                                         sphit.getPosition()[0],
                                                                         sphit.getPosition()[1],
                                                                         sphit.getPosition()[2])
                                                    tg_bremproj.SetPoint(1,
                                                                         sphit.getPosition()[0] - (750 - sphit.getPosition()[2])*sphit.getMomentum()[0]/(4000 - sphit.getMomentum()[2]),
                                                                         sphit.getPosition()[1] - (750 - sphit.getPosition()[2])*sphit.getMomentum()[1]/(4000 - sphit.getMomentum()[2]),
                                                                         750)
                                                    tg_bremproj.SetLineColor(1)
                                                    tg_bremproj.SetLineWidth(3)
                                                    tg_bremproj.Draw("LINE same")

                                photondaughterslist = []
                                for daughters in photondaughters:
                                    for daughter in daughters:
                                        photondaughterslist.append(daughter)

                                d = {}
                                for i in range(len(photondaughterslist)):
                                    d["TGraph2D_{0}".format(i)] = TGraph2D()

                                for it in event.SimParticles_sim:
                                    for i in range(len(photondaughterslist)):
                                        daughter = photondaughterslist[i]
                                        if it.first == daughter:
                                            pdgid = it.second.getPdgID()
                                            vertex = it.second.getVertex()
                                            endVertex = it.second.getEndPoint()
                                            print(evtcount, pdgid, vertex[2], endVertex[2], math.sqrt(it.second.getEnergy()**2 - it.second.getMass()**2))
                                            tg = d["TGraph2D_{0}".format(i)]
                                            tg.SetPoint(0,vertex[0],vertex[1],vertex[2])
                                            tg.SetPoint(1,endVertex[0],endVertex[1],endVertex[2])

                                            if pdgid == 2212:
                                                tg.SetLineColor(2)
                                            elif pdgid == 2112:
                                                tg.SetLineColor(417)
                                            elif pdgid == 211:
                                                tg.SetLineColor(616)
                                            elif pdgid == -211:
                                                tg.SetLineColor(432)
                                            elif pdgid == 22:
                                                tg.SetLineColor(91)
                                            elif pdgid == 130:
                                                tg.SetLineColor(801)
                                            elif pdgid == 310:
                                                tg.SetLineColor(811)
                                            elif pdgid == 321:
                                                tg.SetLineColor(609)
                                            elif pdgid == -321:
                                                tg.SetLineColor(426)
                                            elif pdgid == 11:
                                                tg.SetLineColor(4)

                                            tg.SetLineWidth(2)
                                            tg.Draw("LINE same")

                                h3_vol.GetXaxis().SetTitle("x [mm]")
                                h3_vol.GetYaxis().SetTitle("y [mm]")
                                h3_vol.GetZaxis().SetTitle("z [mm]")

                                c3d.SetTitle(str(evtcount))
                                c3d.Print(EDPath + str(evtcount) + "_hcal.root")

                                del h3_vol
                                del h3_hcal
                                del c3d

                            evtcount += 1
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
    
    parser.add_option('--bdt_path', dest='bdt_path', default='/sdf/home/h/horoho/ldmx/hcal_bdt_weights.pkl', help='BDT model to use')
    parser.add_option('--evtdisplay_path', dest='evtdisplay_path', default='/sdf/home/h/horoho/ldmx/analysis/august23/', help='Where to put events that pass veto')
    parser.add_option('--swdir', dest='swdir', default='/sfs/qumulo/qhome/lbc2qnt/ldmx-sw/install', help='ldmx-sw build directory')
    
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/sdf/group/ldmx/data/mc23/v14/4.0GeV/v3.2.0_ecalPN_tskim-batch3/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/project/hep_aag/ldmx/ap/visibles/produced/mAp_050/', help='name of signal file directory')


    (options, args) = parser.parse_args()


    # load bdt model from pkl file
    gbm = pkl.load(open(options.bdt_path, 'rb'))

    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.bkg_dir, options.evtdisplay_path, True, gbm)
