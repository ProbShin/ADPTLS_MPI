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

class LSFaultMPI{
  public:
    LSFaultMPI(const string&fA, const string&fE, const string& fb, vector<int>&vfts, const string& ftfa_base, const int rk, const int np);
    ~LSFaultMPI(){
      if(Atut) delete Atut;
      if(A4Res) delete A4Res;
      if(E4Res) delete E4Res;
      if(b4Res) delete b4Res;
      if(Eb)    delete []Eb;
    }
  public:
    int cnt()const{ return ftn; }
    int next_position() const { return rstep_pool.empty()?-1:rstep_pool.back(); }
    void boom(MtxSpMPI* &A, double* nby1, double* x_loc, vector<double>&saved_x_loc, double *bloc, double*r_loc, double& rddot_loc, double& rddot, double&beta);


  private:
    int rank, nproc;
    int ftn;
    vector<int>    rstep_pool;
    vector<string> rfA_pool;
    FTMtxMPI *Atut;
  public:
    MtxSpMPI *A4Res;
    MtxDen   *E4Res;
    MtxDen   *b4Res;
    double   *Eb;    //E'*b
};



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

  //FTMtxMPI *Atut_;
  MtxSpMPI *A_;
  MtxDen b_;
  
  LSFaultMPI *faults;


  //vector<int> r_ft_pool_;
  //vector<string> r_ft_fA_pool_;
  vector<double> saved_x_loc_;

protected: string file_A_;
protected: string file_E_;
protected: string file_rhs_;

};


#endif


