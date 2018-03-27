#ifndef MPI_MTX_FTOL_HPP
#define MPI_MTX_FTOL_HPP

#include "mtx_basic.hpp"
using namespace std;

// ============================================================================
// Fault Tolerated System/Matrix
// ----------------------------------------------------------------------------
// if there is no fault
//   -----------
//   |         |    
//   |  org    |     
//   |         |  
//   -----------  
//     
//     ||
//     \/
//
//   -----------
//   |   np0   |
//   -----------
//   |   np1   |
//   -----------
//   |   np2   |
//   -----------
//
//
// if faulte happens
// some row and cols of 
//   ----------- 
//   |         |    
//   |   org   |     
//   |         |    
//   -----------
//
// will be replace with the cols and rows from the
// following matrix
//
//             
//             -------
//             |     |
//             | AE  |
//             |     |
//   -----------------
//   |   E'A   | E'AE|
//   -----------------
//
//   org is an  n_ by n_ matrix
//   AE  is an  n_ by k_ matrix
//   EAE is an  k_ by k_ matrix
//
//   local matrix (e.g. np0) is an nloc_ by k_ matrix
// ============================================================================

class FTMtxMPI : public MtxSpMPI{
public: FTMtxMPI(const string&file_A, const string&file_E, int rnk, int np, int ftn=0); 
virtual ~FTMtxMPI(){ //for(auto x: G) if(x) {delete []x; x=nullptr;}
};

public:
      virtual int get_sys_E_size() const { return G_sys_E_size_; }
      void fault_boom(int ftn);
      void Atut_sub_mtx_times_x_saved(vector<int> &rows, vector<int>&cols, vector<double>&args, double*ftbperrank);
virtual void dump();
private:
virtual void error(const string& s1);
virtual void error(const string& s1,const string&s2);


private: 
//vector<double*>  G;
//vector<int>  Goff;

vector<int>    Gia;
vector<int>    Gja;
vector<double> Ga;

vector<double> GAE, GEA, GEAE;
//unordered_map<int, unordered_map<int,double>> G_;
int G_sys_A_size_;   //n
int G_sys_E_size_;   //k
vector<int> G_nE_rcvcnt_, G_nE_displs_;  //non-even splite

};

#endif


