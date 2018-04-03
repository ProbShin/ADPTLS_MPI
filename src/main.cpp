#include<cstring>
#include<iostream>
#include<cstdio>     /* stderr */
#include<cstdlib>    /* exit */
#include<fstream>    /* fstream */

#include <cmath>     /* fabs */

#include "mpi.h"
#include "def.hpp"
#include "adptls.hpp"
using namespace std;


int main(int argc, char const *argv[]){
  int nproc, rank;
  MPI_Init(&argc, (char***) &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  
  string file_A, file_E, file_rhs, ftfa;
  vector<int> vfts;
 
  if(argc<=1) {printf("usage: check run.sh\n"); MPI_Finalize(); exit(1); }
  for(int i=1; i<argc; i++){
    if(     !strcmp(argv[i], "-fA"  )) file_A = argv[++i];
    else if(!strcmp(argv[i], "-fE"   )) file_E = argv[++i];
    else if(!strcmp(argv[i], "-rhs"  )) file_rhs = argv[++i];
    else if(!strcmp(argv[i], "-fts"))  for(int j=i+1; j<argc && argv[j][0]!='-'; j++, i++) vfts.push_back(atoi(argv[j]));
    else if(!strcmp(argv[i], "-ftbase"))  ftfa = argv[++i];
    else { fprintf(stderr,"Warning! unused input argument\"%s\"",argv[i]); }
  }
  
  
  MPI_ADLS_CG adcg(file_A, file_E, file_rhs, ftfa, vfts, rank, nproc);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  //int tim=-MPI_Wtime();
  adcg.solve();
  //tim+=MPI_Wtime(); if(rank==0) printf("Solver time %f\n", tim);
  
  //if(rank==ROOT_ID){
  //  fprintf(stdout,"\nx(%d):\n",(int)adcg.v_x_.size());
  //  for(auto x : adcg.v_x_) fprintf(stdout, "%lg ", x);
  //  printf(stdout, "\n");
  //

  
  MPI_Finalize();
  return  0;
}





