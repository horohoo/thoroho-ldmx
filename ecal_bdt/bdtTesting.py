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
        
        h0 = TH1F("totalevents", "totalevents", 3, 0, 2)
        h1 = TH1F("passtrigger", "passtrigger", 3, 0, 2)
        h2 = TH1F("ecalEnergyReq", "ecalEnergyReq", 3, 0, 2)
        h3 = TH1F("passTrackerVeto", "passTrackerVeto", 3, 0, 2)
        h4 = TH1F("passBDT", "passBDT", 3, 0, 2)
        
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


                for hit in EcalRecHits:
                    Etot += hit.getEnergy()
                    if hit.getZPos() < 500:
                        Eupstream += hit.getEnergy()
                    else:
                        Edownstream += hit.getEnergy()

                for hit in HcalRecHits:
                    if hit.getZPos() >= 870:
                        EHcal += 12.2*hit.getEnergy()

                h0.Fill(1)
                if Eupstream < 1500:
                    h1.Fill(1)

                if Eupstream < 1500 and Edownstream >= 2500 and EHcal < 2500:
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

                        
                        if bdt.predict(xgb.DMatrix(np.vstack((evt,np.zeros_like(evt))),np.zeros(2)))[0] >= 0.99993:
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
                                print(filename)

                                
                                c3d = TCanvas()
                                h3_vol = TH3F(str(evtcount), str(evtcount), 2, -1500, 1500, 2, -1500, 1500, 2, 0, 3000)
                                h3_ecal = TH3F(str(evtcount), str(evtcount), 125, -250, 250, 125, 250, 250, 60, 200, 800)
                                #h3_hcal = TH3F("hcalhits", "hcalhits", 100, -1500, 1500, 100, -1500, 1500, 60, 0, 3000)
                                for hit in event.EcalRecHits_sim:
                                    #h.Fill(hit.getEnergy())
                                    h3_ecal.Fill(hit.getXPos(), hit.getYPos(), hit.getZPos(), hit.getEnergy())
                                #for hit in event.HcalRecHits_sim:
                                #    h3_hcal.Fill(hit.getXPos(), hit.getYPos(), hit.getZPos(), hit.getEnergy())
                                c3d.cd()
                                h3_vol.Draw("BOX")
                                h3_ecal.Draw("BOX2 Z")
                                #h3_hcal.SetFillColor(2)
                                #h3_hcal.Draw("BOX2 Z")
                                # Find all tracks and draw them
                                #targetSPhitTracks = []
                                targetSPhits = []
                                for sphit in event.TargetScoringPlaneHits_sim:
                                    if sphit.getPosition()[2] > 0:
                                        #targetSPhitTracks.append(sphit.getTrackID())
                                        targetSPhits.append(sphit)
                                photonDaughters = []
                                electrons = []
                            
                                for it in event.SimParticles_sim:
                                    #for SPhitTrack in targetSPhitTracks:
                                    for SPhit in targetSPhits:
                                        if it.first == SPhit.getTrackID():
                                        #if it.first == SPhitTrack:
                                            if it.second.getPdgID() == 22:
                                                if it.second.getEnergy() >= 2500:
                                                    photonDaughters.append(np.array(it.second.getDaughters()))
                                                    
                                            # pick out recoil electron
                                            if it.second.getPdgID() == 11 and 0 in it.second.getParents():
                                                
                                                tg_recoilproj = TGraph2D()
                                                tg_recoilproj.SetPoint(0,
                                                                       it.second.getVertex()[0],
                                                                       it.second.getVertex()[1],
                                                                       it.second.getVertex()[2])
                                                tg_recoilproj.SetPoint(1,
                                                                       it.second.getVertex()[0] + (750 - it.second.getVertex()[2])*it.second.getMomentum()[0]/it.second.getMomentum()[2],
                                                                       it.second.getVertex()[1] + (750 - it.second.getVertex()[2])*it.second.getMomentum()[1]/it.second.getMomentum()[2],
                                                                       750)
                                                tg_recoilproj.SetLineColor(4)
                                                tg_recoilproj.SetLineWidth(3)
                                                tg_recoilproj.Draw("LINE same")
                                                
                                                tg_recoilgammaproj = TGraph2D()
                                                tg_recoilgammaproj.SetPoint(0, 
                                                                      SPhit.getPosition()[0],
                                                                      SPhit.getPosition()[1],
                                                                      SPhit.getPosition()[2])
                                                tg_recoilgammaproj.SetPoint(1,
                                                                      SPhit.getPosition()[0] + (750 - SPhit.getPosition()[2])*(-1)*SPhit.getMomentum()[0]/(4000 - SPhit.getMomentum()[2]),
                                                                      SPhit.getPosition()[1] + (750 - SPhit.getPosition()[2])*(-1)*SPhit.getMomentum()[1]/(4000 - SPhit.getMomentum()[2]),
                                                                      750)
                                                tg_recoilgammaproj.SetLineColor(1)
                                                tg_recoilgammaproj.SetLineWidth(3)
                                                tg_recoilgammaproj.Draw("LINE same")
                                                
                            
                                photonDaughtersList = []
                                for daughters in photonDaughters:
                                    for daughter in daughters:
                                        photonDaughtersList.append(daughter)
                            
                                d = {}
                                for i in range(len(photonDaughtersList)):
                                    d["TGraph2D_{0}".format(i)] = TGraph2D()


                                for it in event.SimParticles_sim:
                                    for i in range(len(photonDaughtersList)):
                                        daughter = photonDaughtersList[i]
                                        if it.first == daughter:
                                            pdgid = it.second.getPdgID()
                                            vertex = it.second.getVertex()
                                            endVertex = it.second.getEndPoint()
                                            print(pdgid, vertex[2], endVertex[2], math.sqrt(it.second.getEnergy()**2 - it.second.getMass()**2))
                                            tg = d["TGraph2D_{0}".format(i)]
                                            tg.SetPoint(0,vertex[0],vertex[1],vertex[2])
                                            tg.SetPoint(1,endVertex[0],endVertex[1],endVertex[2])

                                            if pdgid == 2212:
                                                tg.SetLineColor(2)
                                                # proton = red
                                            elif pdgid == 2112:
                                                tg.SetLineColor(417)
                                                # neutron = green
                                            elif pdgid == 211:
                                                tg.SetLineColor(616)
                                                # pi+ = magenta
                                            elif pdgid == -211:
                                                tg.SetLineColor(432)
                                                # pi- = cyan
                                            elif pdgid == 111:
                                                tg.SetLineColor(51)
                                                # pi0 = purple
                                            elif pdgid == 22:
                                                tg.SetLineColor(91)
                                                # photon = yellow
                                            elif pdgid == 130:
                                                tg.SetLineColor(801)
                                                # Klong = orange
                                            elif pdgid == 310:
                                                tg.SetLineColor(811)
                                                h.Fill(endVertex[2])
                                                # Kshort = lime
                                                #print("K-short in event")
                                            elif pdgid == 321:
                                                tg.SetLineColor(609)
                                                # K+ = off magenta
                                            elif pdgid == -321:
                                                tg.SetLineColor(426)
                                                # K- = off cyan
                                            elif pdgid == 11:
                                                tg.SetLineColor(4)
                                                # electrons = blue
                                            #else:
                                            #    if pdgid < 10000:
                                            #        print(evtcount, pdgid)
                                            tg.SetLineWidth(2)
                                            tg.Draw("LINE same")
                                            
                                
                                h3_vol.GetXaxis().SetTitle("x [mm]")
                                h3_vol.GetYaxis().SetTitle("y [mm]")
                                h3_vol.GetZaxis().SetTitle("z [mm]")
                            
                                c3d.SetTitle(str(evtcount))
                                c3d.Print(EDPath + str(evtcount) + "_ecal.root")
                            
                                del h3_vol
                                del h3_ecal
                                #del h3_hcal
                                del c3d
            """
                                
                            

        
        
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

    parser.add_option('--bdt_path', dest='bdt_path', default='/sdf/home/h/horoho/ldmx/bdt_v6_weights.pkl', help='BDT model to use')
    parser.add_option('--evtdisplay_path', dest='evtdisplay_path', default='/sdf/home/h/horoho/ldmx/test/', help='Where to put events that pass veto')
    parser.add_option('--swdir', dest='swdir', default='/sdf/home/h/horoho/ldmx/ldmx-sw/install', help='ldmx-sw build directory')
    
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/scratch/tgh7hx/ldmx/batch/targetPN/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/sdf/group/ldmx/data/ap_visibles/v14/mAp_005/', help='name of signal file directory')


    (options, args) = parser.parse_args()


    # load bdt model from pkl file
    gbm = pkl.load(open(options.bdt_path, 'rb'))

    print('Loading bkg_file = ', options.bkg_dir)
    bkgContainer = sampleContainer(options.sig_dir, options.evtdisplay_path, False, gbm)
