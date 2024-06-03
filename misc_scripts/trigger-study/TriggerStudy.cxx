#include "Analysis/TriggerStudy.h"
#include "SimCore/Event/SimParticle.h"
#include "DetDescr/HcalID.h"
#include "Framework/NtupleManager.h"
#include "Event/HcalHit.h"
#include "Event/EcalHit.h"
#include "SimCore/Event/SimCalorimeterHit.h"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace ldmx {

    void TriggerStudy::configure(framework::config::Parameters& ps) {
     
      // These values are hard-coded for the v12 detector
      // z position of the start of the back Hcal for the v12 detector
      HcalStartZ_ = ps.getParameter<double>("HcalStartZ_");
        return;
    }

    void TriggerStudy::analyze(const framework::Event& event) {

      //Grab the SimParticle Map
      auto particle_map{event.getMap<int, ldmx::SimParticle>("SimParticles")};

      //Create vector of points at which A' decays
      double ap_energy;
      double ap_decayz;

      //Loop over all SimParticles
      for (auto const& it : particle_map) {
	SimParticle p = it.second;
	int pdgid = p.getPdgID();
	std::vector<int> parents_track_ids = p.getParents();
	for (auto const& parent_track_id: parents_track_ids) {
	  if (parent_track_id == 0) {
	    if (pdgid == 622){
	      ap_energy = p.getEnergy();
	    }
	    // In simulation, the A' track is propagated until it leaves the
	    // world volume (even if it decays earlier), so to get the position
	    // of the A' decay we need to search for the positron in the e+e-
	    // pair production.
	    if (pdgid == -11) {
	      ap_decayz = p.getVertex()[2];
	    }
	  }
	}
      }
      histograms_.fill("ap_energyvsdecayz", ap_energy, ap_decayz);

      // Grab the Ecal Reconstructed Hits
      std::vector<EcalHit> ecalHits = event.getCollection<EcalHit>("EcalRecHits");
      //Grab the Hcal Reconstructed Hits
      std::vector<HcalHit> hcalHits = event.getCollection<HcalHit>("HcalRecHits");

      // variables to store Ecal energy downstream of different z cuts
      double ecalE = 0.; // for no z cut
      double ecalE_downstream_300mm = 0.;
      double ecalE_downstream_400mm = 0.;
      double ecalE_downstream_500mm = 0.;

      double ecalE_upstream_500mm = 0.; // variable for upstream energy trigger

      // variables for Hcal energy
      double hcalE = 0.;
      double hcalE_downstream_500mm = 0.;      
      float xi = 12.16; // scaling factor for Hcal hits

      // variables for Ecal and Hcal energy sums
      double ecalhcalE = 0.;
      double ecalhcalE_downstream_500mm = 0.;
      
      for(const EcalHit &hit : ecalHits) {
	float z_hit = hit.getZPos();
	float energy = hit.getEnergy();
	ecalE += energy;
	if (z_hit >= 300) {
	  ecalE_downstream_300mm += energy;
	}
	if (z_hit >= 400) { 
	  ecalE_downstream_400mm += energy;
	}
	if (z_hit >= 500) {
	  ecalE_downstream_500mm += energy;
	}
	if (z_hit < 500) {
	  ecalE_upstream_500mm += energy;
	}
      }
      for (const HcalHit &hit : hcalHits) {
	float z_pos = hit.getZPos();
	float energy = xi * hit.getEnergy();
	hcalE += energy;
	    
	if(z_pos >= 500){
	  hcalE_downstream_500mm += energy;
	}
      }
	  
      ecalhcalE = ecalE + hcalE;
      ecalhcalE_downstream_500mm = ecalE_downstream_500mm + hcalE_downstream_500mm;
      histograms_.fill("decayz_all", ap_decayz);
      
      if(ecalE_downstream_500mm >= 100){
	histograms_.fill("decayz_downstream_500mm_100MeV", ap_decayz);
      }
      if(ecalE_downstream_500mm >= 1500){
	histograms_.fill("decayz_downstream_500mm_1500MeV", ap_decayz);
      }
      if(ecalE_downstream_500mm >= 2000){
	histograms_.fill("decayz_downstream_500mm_2000MeV", ap_decayz);
      }
      if(ecalE_downstream_400mm >= 2500){
	histograms_.fill("decayz_downstream_400mm", ap_decayz);
      }
      if(ecalE_downstream_500mm >= 2500){
	histograms_.fill("decayz_downstream_500mm", ap_decayz);
      }
      if(ecalE_upstream_500mm <= 1500){
	histograms_.fill("decayz_upstream_500mm", ap_decayz);
      }
      if(ecalE_downstream_300mm >= 2500){
	histograms_.fill("decayz_downstream_300mm", ap_decayz);
      }
      if(ecalE >= 2500){
	histograms_.fill("decayz_allEcal_2500MeV", ap_decayz);
      }
      if(ecalE <= 1500){
	histograms_.fill("decayz_allEcal_1500MeV", ap_decayz);
      }
      if(ecalE_downstream_500mm >= 2500 || ecalE < 1500){
	histograms_.fill("decayz_visplusinvis", ap_decayz);
      }
      if(ecalE_downstream_500mm >= 3000){
	histograms_.fill("decayz_downstream_500mm_3000MeV", ap_decayz);
      }
      if(ecalhcalE >= 2500){
	histograms_.fill("decayz_ecalhcal_2500MeV", ap_decayz);
      }
      if(ecalhcalE_downstream_500mm >= 2500){
	histograms_.fill("decayz_ecalhcal_500mm_2500MeV", ap_decayz);
      }

      // loop for downstream trigger rate
      for (int i=0; i < 100; i++) {
	double zmin = 10.*i;
	double Edownstream = 0.;
	for (const EcalHit &hit : ecalHits) {
	  if (hit.getZPos() > zmin) {
	    Edownstream += hit.getEnergy();
	  }
	}
	if (Edownstream > 100) {
	  histograms_.fill("trigger_downstream_100MeV", zmin);
	}
	if (Edownstream > 1500) {
	  histograms_.fill("trigger_downstream_1500MeV", zmin);
	}
	if (Edownstream > 2000) {
	  histograms_.fill("trigger_downstream_2000MeV", zmin);
	}
	if (Edownstream > 2500) {
	  histograms_.fill("trigger_downstream_2500MeV", zmin);
	}
	if (Edownstream > 3000) {
	  histograms_.fill("trigger_downstream_3000MeV", zmin);
	}
      }

      // loop for upstream trigger rate
      // this is the trigger we decided on for the analysis
      for (int i=0; i < 100; i++) {
        double zmax = 10.*i;
        double Eupstream = 0.;
        for (const EcalHit &hit : ecalHits) {
	  if (hit.getZPos() < zmax) {
	    Eupstream += hit.getEnergy();
	  }
	}
	if (Eupstream < 1000) {
	  histograms_.fill("trigger_upstream_1000MeV", zmax);
	}
	if (Eupstream < 1500) { // this is our trigger
	  histograms_.fill("trigger_upstream_1500MeV", zmax);
	}
	if (Eupstream < 2000) {
	  histograms_.fill("trigger_upstream_2000MeV", zmax);
	}
	if (Eupstream < 2500) {
	  histograms_.fill("trigger_upstream_2500MeV", zmax);
	}
      }

        return;
    }

    void TriggerStudy::onFileOpen(framework::EventFile&) {

        return;
    }

    void TriggerStudy::onFileClose(framework::EventFile&) {

        return;
    }

    void TriggerStudy::onProcessStart() {

        return;
    }

    void TriggerStudy::onProcessEnd() {

        return;
    }
}

DECLARE_ANALYZER_NS(ldmx, TriggerStudy)
