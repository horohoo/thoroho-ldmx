#!/bin/python

#this script is used for testing specific sets of deepbrem biasing conditions on various seeds

import os
import sys
import json

from LDMX.Framework import ldmxcfg

# set a 'pass name';
thisPassName="sim"
p=ldmxcfg.Process(thisPassName)

p.maxTriesPerEvent = 1 

b_thresh = 4800.0 
p_sorter =  3160.0 

#import all processors

from LDMX.Biasing import filters
from LDMX.Biasing import util
from LDMX.Biasing import include as biasinclude
from LDMX.SimCore import generators
from LDMX.SimCore import simulator
from LDMX.SimCore import bias_operators

detector='ldmx-det-v14-8gev'
sim = simulator.simulator("deepBrem")
sim.setDetector(detector, True) # True turns on scoring planes
sim.generators.append(generators.single_8gev_e_upstream_tagger())
# below include all the filters we want

sim.actions.extend([
   util.PartialEnergySorter(p_sorter),
   filters.TaggerVetoFilter(thresh=7600.),
   filters.PrimaryToEcalFilter(50.0),
   filters.DeepEcalProcessFilter(bias_threshold=b_thresh, processes=["conv","phot)"], ecal_min_Z=500.0, require_photon_fromTarget=True)
   ])
sim.beamSpotSmear = [20.,80.,0.]
sim.description = 'Inclusive sample with deep brem filter'


############################################################
# Below should be the same for all sim scenarios
#
#Set run parameters. These are all pulled from the job config
#

p.run = int(sys.argv[1])
nElectrons = 1
beamEnergy = 4.0; #in GeV

p.maxEvents = 1000000

#p.histogramFile = f'hist.root'
p.outputFiles = [sys.argv[2]]

import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Hcal.HcalGeometry
import LDMX.Hcal.hcal_hardcoded_conditions

from LDMX.Ecal import digi as eDigi
from LDMX.Ecal import vetos
from LDMX.Hcal import digi as hDigi
from LDMX.Hcal import hcal

from LDMX.Recon.electronCounter import ElectronCounter


#calorimeters
ecalDigi = eDigi.EcalDigiProducer('ecalDigi')
ecalReco = eDigi.EcalRecProducer('ecalRecon')
ecalVeto = vetos.EcalVetoProcessor('ecalVetoBDT')


# electron counter so simpletrigger doesn't crash
eCount = ElectronCounter(1, "ElectronCounter") # first argument is number of electrons in simulation
eCount.use_simulated_electron_number = True
eCount.input_pass_name=thisPassName


p.sequence=[sim,
            ecalDigi,
            ecalReco,
            ecalVeto,
            eCount
            ]

p.keep = ["drop MagnetScoringPlaneHits", "drop TrackerScoringPlaneHits", "drop HcalScoringPlaneHits"]

#p.outputFiles= [f'ldmx-analysis/dbc_validation/combined_factor_testing/visible/partial_energy/outs/400min_{p.run}_evts.root']  #[sys.argv[2]]
#p.histogramFile = f'ldmx-analysis/dbc_validation/combined_factor_testing/visible/partial_energy/outs/400min_{p.run}_hist.root'

p.termLogLevel = 2
logEvents=20
if p.maxEvents < logEvents :
   logEvents = p.maxEvents
p.logFrequency = int(p.maxEvents/logEvents)

#from LDMX.DQM import dqm
#p.sequence.append(dqm.SampleValidation())
#p.sequence.extend(dqm.ecal_dqm)
