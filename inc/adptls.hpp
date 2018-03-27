#ifndef MPI_ADLS_HPP
#define MPI_ADLS_HPP

#include <string>
#include <vector>
#include "mpi.h"
#include "def.hpp"
#include "ftmtx.hpp"
#include "mtx_basic.hpp"


using namespace std;
// ============================================================================
// MPI version of Adaptive Liner Solver (CG)
// ============================================================================
class MPI_ADLS_CG {
public: MPI_ADLS_CG(const string &f_A, const string& f_E, 
            const string& f_rhs, 
            const string& ftfa_base,
            vector<int>&vfts,
            int rank,int nproc);

public: virtual ~MPI_ADLS_CG();
public: 
  MPI_ADLS_CG(MPI_ADLS_CG &&)=delete;
  MPI_ADLS_CG(const MPI_ADLS_CG &)=delete;
  MPI_ADLS_CG& operator=(MPI_ADLS_CG&&)=delete;
  MPI_ADLS_CG& operator=(const MPI_ADLS_CG&)=delete;

public: void   solve();
public: void   xloc2xglb(int nloc, double* xloc, int* rcvcnt, int* displs, double*x, int ftn, int ftn_loc, double* saved_xloc, MtxDen*E);
public: double get_rtol_from_xloc(double *xloc, MtxSpMPI *A, int ftn, int ftn_loc, double* saved_xloc, MtxDen*E, MtxDen*b);


public:  virtual void dump();
private: virtual void error(const string&s1, const string&s2);
private: virtual void error(const string&s1);

public:
  const int rank_;
  const int nproc_;

  // CG's variable
  //vector<double> v_p_;        // p global
  vector<double> v_p_loc_;    // p local
  vector<double> v_Ap_loc_;   //Ap local
  vector<double> v_x_;        // x global
  vector<double> v_x_loc_;    // x local
  vector<double> v_r_loc_;    //res local

  FTMtxMPI *Atut_;
  MtxSpMPI *A_;
  MtxDen b_;

  vector<int> r_ft_pool_;
  vector<string> r_ft_fA_pool_;
  vector<double> saved_x_loc_;

protected: string file_A_;
protected: string file_E_;
protected: string file_rhs_;

};


#endif


