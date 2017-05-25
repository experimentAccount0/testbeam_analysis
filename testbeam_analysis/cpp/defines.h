#ifndef DEFINES_H
#define DEFINES_H

//#pragma pack(1) //data struct in memory alignement
#pragma pack(push, 1)

// for (u)int64_t event_number
#ifdef _MSC_VER
#include "external/stdint.h"
#else
#include <stdint.h>
#endif

//structure of the hits
typedef struct HitInfo{
  uint64_t eventNumber;   //event number value (unsigned long long: 0 to 18,446,744,073,709,551,615)
  unsigned char frame; //relative BCID value (unsigned char: 0 to 255)
  unsigned short int column;       //column value (unsigned int: 0 to 255)
  unsigned short int row;     //row value (unsigned short int: 0 to 65.535)
  unsigned short int charge;          //tot value (unsigned int: 0 to 255)
} HitInfo;

//structure to store the hits with cluster info
typedef struct ClusterHitInfo{
  uint64_t eventNumber;   //event number value (unsigned long long: 0 to 18,446,744,073,709,551,615)
  unsigned char frame; //relative BCID value (unsigned char: 0 to 255)
  unsigned short int column;       //column value (unsigned char: 0 to 255)
  unsigned short int row;     //row value (unsigned short int: 0 to 65.535)
  unsigned short int charge;          //tot value (unsigned char: 0 to 255)
  unsigned short clusterID;	  //the cluster id of the hit
  unsigned char isSeed;	  	  //flag to mark seed pixel
  unsigned short clusterSize; //the cluster size of the cluster belonging to the hit
  unsigned short nCluster;	  //the number of cluster in the event
} ClusterHitInfo;

//structure to store the cluster
typedef struct ClusterInfo{
  uint64_t eventNumber;  	  //event number value (unsigned long long: 0 to 18,446,744,073,709,551,615)
  unsigned short ID;	  	  //the cluster id of the cluster
  unsigned short n_hits; 		  //sum tot of all cluster hits
  float charge; 	  //sum charge of all cluster hits
  unsigned short int seed_column;  //seed pixel column value (unsigned char: 0 to 255)
  unsigned short int seed_row;//seed pixel row value (unsigned short int: 0 to 65.535)
  float mean_column;		  //column value (unsigned short int: 0 to 65.535)
  float mean_row;			  //row value (unsigned short int: 0 to 65.535)
  float err_column; 		  //column position error
  float err_row; 		  //row position error
} ClusterInfo;

#pragma pack(pop)  // pop needed to suppress VS C4103 compiler warning
#endif // DEFINES_H
