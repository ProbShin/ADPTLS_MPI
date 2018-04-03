/*************************************************************************
    > File Name: main.cpp
    > Author: xc
    > Descriptions: 
    > Created Time: Mon 02 Apr 2018 12:31:44 PM EDT
 ************************************************************************/
#include<random>
#include<chrono>
#include<string>
#include<iostream>
#include<fstream>
#include<cstring>
#include "mpi.h"
#include "def.hpp"
#include "mtx_basic.hpp"

using namespace std;

void mm_write_den_vector(const string& file_name, double *dat, int size);


int main(int argc, char const *argv[]){
  int nproc, rank;
  MPI_Init(&argc, (char***) &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  string file_A;
  string file_b;
  unsigned int rseed=std::chrono::system_clock::now().time_since_epoch().count();

  if(argc<=1) {
    fprintf(stderr,"\nusage: [mpirun -np <np>] %s -fA <mtxA> -fb <mtxb> [-s <rand seed>]\n",argv[0]); 
    MPI_Finalize(); 
    exit(1);
  }

  for(int i=1; i<argc; i++) {
    if( !strcmp(argv[i], "-fA") ) file_A=argv[++i];
    else if( !strcmp(argv[i], "-fb") ) file_b=argv[++i];
    else if( !strcmp(argv[i], "-s" ) ) rseed =(unsigned int)atoi(argv[++i]);
    else{
      fprintf(stderr, "Warning! unused input argument \"%s\"\n", argv[i]);
    }
  }

  if(file_A=="") {
    fprintf(stderr, "file name of Mtx A is empty!\n"); 
    MPI_Finalize();
    exit(1);
  }

  if(file_b=="") {
    fprintf(stderr, "file name of Mtx b is empty!\n");
    MPI_Finalize();
    exit(1);
  }


  MtxSpMPI *A;
  if(file_A.substr(file_A.size()-5,4)==".mtx")
    A = new MtxSpMPI(file_A, "mm", rank, nproc);
  else
    A = new MtxSpMPI(file_A, "binary", rank, nproc);


  
  int n = A->rows();
  if(A->cols()!=n) {
    fprintf(stderr, "Matrix A(%s) num_cols(%d)!=num_rows(%d)\n",file_A.c_str(), A->cols(), A->rows());
    MPI_Finalize();
    exit(1);
  }




  int nloc = A->rows_loc();
  int * displs = &(A->get_row_displs()[0]);
  int * rcvcnt = &(A->get_row_rcvcnt()[0]);
  
  double *x = new double[n];
  double *xloc=new double[nloc];
  double *bloc = new double[nloc];
  if(!x) {
    fprintf(stderr, "cannot assign enough memory for size(x)=%d\n",n); 
    MPI_Finalize();
    exit(1);
  }
  if(!xloc) {
    fprintf(stderr, "cannot assign enough memory for size(xloc)=%d\n",nloc); 
    MPI_Finalize();
    exit(1);
  }
  if(!bloc) {
    fprintf(stderr, "cannot assign enough memory for size(bloc)=%d\n",nloc); 
    MPI_Finalize();
    exit(1);
  }
  

  if(rank==0){
    std::default_random_engine generator(rseed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for(int i=0; i<n; i++)
      x[i] = distribution(generator);
  }

  MPI_Scatterv(x, rcvcnt, displs, MPI_DOUBLE, xloc, nloc, MPI_DOUBLE,0, MPI_COMM_WORLD);

  A->MultiplyVectorMPI(nloc, xloc, x, bloc);
  
  if(x) delete []x; x=nullptr;
  if(xloc) delete []xloc; xloc=nullptr;

  double *b=nullptr;
  if(rank==0) {
    b = new double[n];
  }

  MPI_Gatherv(bloc, nloc, MPI_DOUBLE, b, rcvcnt, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(bloc) delete[]bloc; bloc=nullptr;
  if(rank==0) mm_write_den_vector(file_b, b, n);
  
  if(b) delete[]b; b=nullptr;
  if(A) delete A; A=nullptr;

  MPI_Finalize();
  return 0;
}



void mm_write_den_vector(const string& file_name, double *dat, int size){
  ofstream file(file_name, std::ofstream::out);
  file<<"%%%%MatrixMarket matrix array real general\n"<<size<<" 1\n";
  for(int i=0;i<size; i++){
    file<<dat[i]<<endl;
  }
  file.close();
}



