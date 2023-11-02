#include "Hcal/HcalNewClusterProducer.h"

#include <exception>
#include <iostream>

#include "Hcal/MyClusterWeight.h"
#include "Hcal/TemplatedClusterFinder.h"
#include "Hcal/WorkingCluster.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"

#include <string>
#include <list>
#include <set>
#include <vector>
#include <queue>

namespace hcal {

  HcalNewClusterProducer::HcalNewClusterProducer(const std::string& name,
						 framework::Process& process) : Producer(name, process) {}

  void HcalNewClusterProducer::configure(framework::config::Parameters& parameters) {
    EminSeed_ = parameters.getParameter<double>("EminSeed");
    EnoiseCut_ = parameters.getParameter<double>("EnoiseCut");
    deltaTime_ = parameters.getParameter<double>("deltaTime");
    deltaR_ = parameters.getParameter<double>("deltaR");
    EminCluster_ = parameters.getParameter<double>("EminCluster");
    cutOff_ = parameters.getParameter<double>("cutOff");
    clusterCollName_ = parameters.getParameter<std::string>("clusterCollName");
  }

  static bool compHitTimes(ldmx::HcalHit a, ldmx::HcalHit b) {
    return a.getTime() < b.getTime();
  }


  // add a strip neighbor
  void HcalNewClusterProducer::AddBackStripNeighbor(int id1, int id2) {
    if( back_strip_neighbors.count(id1) ) back_strip_neighbors[id1].push_back(id2);
    else back_strip_neighbors[id1]={id2};
    if( back_strip_neighbors.count(id2) ) back_strip_neighbors[id2].push_back(id1);
    else back_strip_neighbors[id2]={id1};
  }

  // add a layer neighbor
  void HcalNewClusterProducer::AddBackLayerNeighbor(int id1, int id2) {
    if( back_layer_neighbors.count(id1) ) back_layer_neighbors[id1].push_back(id2);
    else back_layer_neighbors[id1]={id2};
    if( back_layer_neighbors.count(id2) ) back_layer_neighbors[id2].push_back(id1);
    else back_layer_neighbors[id2]={id1};
  }

  // fill the lists of neighbours for back hcal
  void HcalNewClusterProducer::MakeBackNeighborMap(std::map<ldmx::HcalID, TVector3> positionMap){
    for (auto hit1 = positionMap.begin(); hit1 != positionMap.end(); hit1++) {
      for (auto hit2 = hit1; hit2 != positionMap.end(); hit2++) {
        if (hit1 == hit2) continue;
        auto &id1 = hit1->first;
        auto &id2 = hit2->first;

        // select only IDs with same section
        if (id1.section() != id2.section()) continue;

        // find adjacent strips in the same layer
        if (id1.layer() == id2.layer()) {
          if (fabs(id1.strip() - id2.strip()) == 1) {
            AddBackStripNeighbor(id1.raw(), id2.raw());
          }
        }

        // find adjacent strips in adjacent layers
        if (fabs(id1.layer() - id2.layer()) == 1) {
          AddBackLayerNeighbor(id1.raw(), id2.raw());
        }

      }
    }
  }

  TVector3 HcalNewClusterProducer::getPosition(std::map<ldmx::HcalID, TVector3> positionMap, ldmx::HcalID thisHit){
    std::map<ldmx::HcalID, TVector3>::iterator it  = positionMap.find(thisHit);
    TVector3 hitPos = it->second;
  }

  void HcalNewClusterProducer::make2DClusters(framework::Event& event, std::vector<ldmx::HcalCluster>& hcalClusters){
    // get geometry
    auto hcalGeom = getCondition<ldmx::HcalGeometry>(ldmx::HcalGeometry::CONDITIONS_OBJECT_NAME);
    hcalGeom.buildStripPositionMap();
    std::map<ldmx::HcalID, TVector3> positionMap = hcalGeom.getStripPositionMap();
    // make maps of ID to layer and strip neightbors
    MakeBackNeighborMap(positionMap);
    // make seed list
    std::list<const ldmx::HcalHit*> seedList;

    // extract hits
    std::vector<ldmx::HcalHit> hcalHits =
      event.getCollection<ldmx::HcalHit>("HcalRecHits");

    if (hcalHits.empty()) {
      return;
    }

    // make hit iterator vector
    std::vector<int> hits;
    hits.reserve(hcalHits.size());

    // energy cut
    for (unsigned int i=0;i<hcalHits.size();++i) {
      if (hcalHits[i].getEnergy() > EnoiseCut_) hits.emplace_back(i);
      //std::cout<<" hit "<<i<<" with position "<<hcalHits[i].getXPos()<<" , "<< hcalHits[i].getYPos() << " , " << hcalHits[i].getZPos()<<std::endl;
    }

    // sort by time
    if (hcalHits.size() > 0) {
      std::sort(hcalHits.begin(), hcalHits.end(), compHitTimes);
    }

    // loop over indices of hits in this event
    auto itSeed = hits.begin();
    while (itSeed != hits.end())
      {

	// take this hit as a seed (hits order in time now)
	const ldmx::HcalHit& hitSeed = hcalHits[*itSeed];

	// find section:
	//if(hitSeed.section() != ldmx::HcalID::HcalSection::BACK) continue

	// check energy, if less iterate
	if (*itSeed==-1 or hitSeed.getEnergy() < EminCluster_) {
	  ++itSeed;
	  continue;
	}
	//std::cout<<"energy "<<hitSeed.getEnergy()<<" with cut at < "<<EminCluster_<<std::endl;
	double timeStart = hitSeed.getTime();
	//std::cout<<" time "<< timeStart << "<"<< deltaTime_<<std::endl;
	//find the range around the seed time to search for other hits to form clusters
	auto iterStart(itSeed), iterStop(itSeed);
	while (iterStop  != hits.end()   and (*iterStop==-1  or hcalHits[*iterStop].getTime() - timeStart < deltaTime_))  ++iterStop;
	while (iterStart != hits.begin() and (*iterStart==-1 or timeStart - hcalHits[*iterStart].getTime() < deltaTime_)) --iterStart;
	++iterStart;

	// get the section, layer, strip IDs:
	//std::cout<<" Hit ID: section  "<<hcalHits[*itSeed].getSection()
	//<<" layer "<<hcalHits[*itSeed].getLayer()
	//  <<" strip "<<hcalHits[*itSeed].getStrip()
	//  <<std::endl;

	//get cuttent ID
	ldmx::HcalID id(hitSeed.getID());

	// queue of unique IDs
	std::queue<ldmx::HcalID> idToVisit;
	// map of ID to true/false if already checked for hits
	std::map<int, bool> isVisited;

	// put the seed as first hit in the cluster list
	std::vector<int> clusterList{*itSeed};
	ldmx::HcalID seedId = hcalHits[*itSeed].getID();
	idToVisit.push(seedId);
	*itSeed=-1;

	// loop until no more id's to visit
	while (!idToVisit.empty())
	  {
	    // get first ID from the queue
	    ldmx::HcalID visitId = idToVisit.front();
	    isVisited[visitId.raw()]=true;

	    // get lists of neightbors of strips and layers
	    std::vector<int>  strip_neighborsId = back_strip_neighbors.find(seedId.raw())->second;
	    std::vector<int>  layer_neighborsId = back_layer_neighbors.find(seedId.raw())->second;

	    //loop through layer neighbours
	    for (const auto& ilayerId : layer_neighborsId){

	      if (isVisited[ilayerId]) continue;
	      isVisited[ilayerId] = true;

	      // loop over other hits in list, add to cluster if in these neighbor layers
	      for (auto it=iterStart; it != iterStop; ++it)
		{
		  if (*it==-1) continue;
		  if (hcalHits[*it].getID() != ilayerId) continue;

		  if (hcalHits[*it].getEnergy() > EminSeed_) idToVisit.push(ilayerId);
		  clusterList.push_back(*it);
		  *it = -1;
		}

	    }
	    // loop separately through strip neightbors
	    for (const auto& istripId : strip_neighborsId){

	      if (isVisited[istripId]) continue;
	      isVisited[istripId] = true;

	      //loop over the hcalHits, check if one is in the neighbor list and add it to the cluster
	      for (auto it=iterStart; it != iterStop; ++it)
		{
		  if (*it==-1) continue;
		  if (hcalHits[*it].getID() != istripId) continue;

		  if (hcalHits[*it].getEnergy() > EminSeed_) idToVisit.push(istripId);
		  clusterList.push_back(*it);
		  *it = -1;
		}

	    }
	    idToVisit.pop();
          }

	auto functorEnergy = [&hcalHits](int a, int b) { return hcalHits[a].getEnergy() > hcalHits[b].getEnergy(); };
	std::sort(clusterList.begin(),clusterList.end(),functorEnergy);
	fillClusters(hcalHits, clusterList, hcalClusters);
	++itSeed;
      }

  }

  void HcalNewClusterProducer::fillClusters(const std::vector<ldmx::HcalHit>& hcalHits, const std::vector<int>& clusterList, std::vector<ldmx::HcalCluster>& hcalClusterList){

    double totalEnergy(0), xCOG(0), yCOG(0), zCOG(0);
    ldmx::HcalCluster cluster;
    for (auto idx : clusterList)
      {
	ldmx::HcalID hcalid        = hcalHits[idx].getID();
        totalEnergy    += hcalHits[idx].getEnergy();
        xCOG           += hcalHits[idx].getXPos()*hcalHits[idx].getEnergy();
        yCOG           += hcalHits[idx].getYPos()*hcalHits[idx].getEnergy();
        zCOG           += hcalHits[idx].getZPos()*hcalHits[idx].getEnergy(); //TODO - 3D position shouldnt work like this!

      }


    const ldmx::HcalHit& seedHit = hcalHits[clusterList[0]];
    double time            = seedHit.getTime();


    if(totalEnergy !=0){
      xCOG /= totalEnergy;
      yCOG /= totalEnergy;
      zCOG /= totalEnergy;
      }
    cluster.setEnergy(totalEnergy);
    cluster.setTime(time);
    cluster.setCentroidXYZ(xCOG, yCOG, zCOG);
    cluster.setNHits(hcalHits.size());
    hcalClusterList.emplace_back(cluster);

  }

  std::vector<ldmx::HcalCluster> HcalNewClusterProducer::finalCluster(std::vector<ldmx::HcalCluster>& hcalClusterList){

    int iters = 0;

    std::vector<ldmx::HcalCluster> oldClusterList = hcalClusterList;
    std::vector<ldmx::HcalCluster> newClusterList;
    std::vector<ldmx::HcalCluster> finalClusterList;

    while (newClusterList.size() != oldClusterList.size()) {
      if (iters > 0) {
	oldClusterList = newClusterList;
	newClusterList.clear();
      }
      iters++;

    //combine clusters within a certain distance
    //std::vector<ldmx::HcalCluster> finalclusterList;
    std::vector<int> clustersUsed;
    //loop over the list of clusters
    for (unsigned int i = 0; i < oldClusterList.size(); i++)
      {
	double x1          = oldClusterList[i].getCentroidX();
	double y1          = oldClusterList[i].getCentroidY();
	double z1          = oldClusterList[i].getCentroidZ();
	double e1          = oldClusterList[i].getEnergy();
	ldmx::HcalCluster bigCluster;
	double combinedX = x1*e1;
	double combinedY = y1*e1;
	double combinedZ = z1*e1;
	double combinedE = e1;
	int nCombined = 0;

	std::cout << "cluster " << i << " with position " << x1 << " , " << y1 << " , " << z1 << std::endl;

	if(i !=0 and std::count(clustersUsed.begin(), clustersUsed.end(), i)){
	  //std::cout << "cluster already used" << std::endl;
	  continue;
	}
	//loop again
	for (unsigned int j = i; j < oldClusterList.size(); j++)
	  {
	    // do not combine the same cluster
	    if(i == j) continue;
	    double x2          = oldClusterList[j].getCentroidX();
	    double y2          = oldClusterList[j].getCentroidY();
	    double z2          = oldClusterList[j].getCentroidZ();
	    double e2          = oldClusterList[j].getEnergy();
	    // find cluster difference
	    double r12 = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
	    std::cout<<"Distance "<<sqrt(r12)<<std::endl;
	    //combine if close:
	    if(sqrt(r12) < deltaR_){
	      combinedX += x2*e2;
	      combinedY += y2*e2;
	      combinedZ += z2*e2;
	      combinedE += e2;
	      nCombined += 1;
	        std::cout<<" combined in loop "<<nCombined<<" clusters "<<std::endl;
	      clustersUsed.push_back(j);
	    }
	  }
	  std::cout<<" combined "<<nCombined<<" clusters "<<std::endl;
	// if we have found close by clusters add the combined cluster:
	if(nCombined != 0) {
	      std::cout<<"adding combined "<<std::endl;
	      bigCluster.setCentroidXYZ(combinedX/combinedE, combinedY/combinedE, combinedZ/combinedE);
	  bigCluster.setTime(1);
	  bigCluster.setEnergy(combinedE);
	  newClusterList.push_back(bigCluster);
	}
	// if not then this cluster remains as its own cluster:
	if(nCombined == 0) {
	    std::cout<<"adding old cluster"<<std::endl;
	  newClusterList.push_back(oldClusterList[i]);
	}
      }
    }
    finalClusterList = newClusterList;
      std::cout<<"final clusters "<<finalClusterList.size()<<std::endl;
    return finalClusterList;
  }

  void HcalNewClusterProducer::produce(framework::Event& event) {
    //std::cout<<"Settings "<<"EminSeed = "<<EminSeed_<<" EnoiseCut = "<<EnoiseCut_<<" deltaTime = "<<deltaTime_<<" EminCluster = "<<EminCluster_<<std::endl;
    std::vector<ldmx::HcalCluster> hcalClusters;
    make2DClusters(event, hcalClusters);
    
    std::cout<<hcalClusters.size()<<std::endl;
    std::vector<ldmx::HcalCluster> finalClusters = finalCluster(hcalClusters);
    event.add(clusterCollName_, finalClusters);
  }

}  // namespace hcal
DECLARE_PRODUCER_NS(hcal, HcalNewClusterProducer);
