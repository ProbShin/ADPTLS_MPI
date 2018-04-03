#ifndef MPI_MTX_BASIC_HPP
#define MPI_MTX_BASIC_HPP
#include <mkl.h>

#include "mpi.h"
#include "mmio.h"
#include <vector>
#include <utility> //pair
#include <unordered_map>
#include <algorithm> //sort
#include <iostream>

using namespace std;

// ============================================================================
// Sparse Symmetric Matrix, stored both upper and lower part 
// ============================================================================
class MtxSp{
public: MtxSp(){}; 
public: MtxSp(const string&file_name);
virtual ~MtxSp(){
  //fprintf(stderr,"~MtxSp want to delete itself\n");
};
        MtxSp(MtxSp&&)=delete;
        MtxSp(const MtxSp&)=delete;
        MtxSp& operator=(MtxSp&&)=delete;
        MtxSp& operator=(const MtxSp&)=delete;

public:
  virtual int  rows() const { return num_rows_; }
  virtual int  cols() const { return num_cols_; }
  virtual void rows(int r)  { num_rows_=r;      }
  virtual void cols(int c)  { num_cols_=c;      }
  vector<int>&    get_ia(){ return ia_; }
  vector<int>&    get_ja(){ return ja_; }
  vector<double>& get_a() { return a_;  }

  //double*  get_aptr()      { return &a_[0]; }
  //MKL_INT* get_mkl_iaptr() { return &MKL_ia_[0]; }
  //MKL_INT* get_mkl_japtr() { return &MKL_ja_[0]; }

  void MultiplyVector(int nrows, double* x, double *y);             
  void MultiplyMatrix(int nrows, int ncols, double* x, double *y);     
  void TransMultiplyVector(int nloc, double* x, double *y);

public:virtual void dump();
private:
virtual void error(const string& s1);
virtual void error(const string& s1, const string&s2);

protected:
vector<int> ia_;
vector<int> ja_;
vector<double> a_;
//vector<MKL_INT> MKL_ia_;
//vector<MKL_INT> MKL_ja_;


private:
int num_rows_;
int num_cols_;
};

// ============================================================================
// MPI Distributed Matrix
// ============================================================================
class MtxSpMPI : public MtxSp {
public: MtxSpMPI(int rnk, int np):MtxSp(),rank_(rnk), nproc_(np) {}
public: MtxSpMPI(const string&file_name,const string&fmt, int rnk, int np); 
virtual ~MtxSpMPI(){
  //row_rcvcnt_.clear();
  //row_displs_.clear();
  //fprintf(stderr,"~MtxSpMPI want to delete itself\n");
};
        MtxSpMPI(MtxSpMPI&&)=delete;
        MtxSpMPI(const MtxSpMPI&)=delete;
        MtxSpMPI& operator=(MtxSpMPI&&)=delete;
        MtxSpMPI& operator=(const MtxSpMPI&)=delete;
public:
  void MultiplyVectorMPI(int nrows, double* xloc, double *xglb_workspace, double *yglb);
  void MultiplyMatrix_Allgatherv(int nrows, int K, double* xglb, double *yloc, double *yglb, int* recvcounts, int* displs);

  
public:
  virtual int rows() const { return num_rows_glb_; }
  virtual int cols() const { return num_cols_glb_; }
  virtual int rows_loc() const { return num_rows_loc_; }
  virtual int cols_loc() const { return num_cols_loc_; }
  
  vector<int>& get_row_displs(){ return row_displs_; }
  vector<int>& get_row_rcvcnt(){ return row_rcvcnt_; }

private: 
  void do_read_mm_mtx_into_G(const string&file_name, unordered_map<int, vector<pair<int, double>>>&G, int &r, int& c);
  void do_read_binary_mtx_into_CSR(const string&file_name, int&rglb, int&cglb, int&rloc, int &cloc, vector<int>&rowcnt, vector<int>&rowpls, const int rank, const int nproc);
public:  virtual void dump();
private: virtual void error(const string& s1);
private: virtual void error(const string& s1,const string& s2);



protected:
const int rank_;
const int nproc_;

vector<int> row_rcvcnt_;                     // nloc for each process
vector<int> row_displs_;                     // used for loc_row_id -> glb_row_id


protected:
int num_rows_loc_, num_cols_loc_;            // matrix size
int num_rows_glb_, num_cols_glb_;            // matrix size


// matrix times vector needed variables
private:  
vector<int>    mv_x_sndidx_,  mv_x_rcvidx_;  // used for pack, unpack
vector<double> mv_x_sndbuf_,  mv_x_rcvbuf_;  // used by MPI_Alltoall
vector<int>    mv_x_sndcnt_,  mv_x_rcvcnt_;  // used by MPI_Alltoall
vector<int>    mv_x_sndpls_,  mv_x_rcvpls_;  // used by MPI_Alltoall

};

// ============================================================================
// Dense Matrix
// ============================================================================
class MtxDen{
public: MtxDen(const string& file_in);
public: virtual ~MtxDen(){};
public:
        MtxDen(MtxDen &&)=delete;
        MtxDen(const MtxDen &)=delete;
        MtxDen& operator=(MtxDen&&)=delete;
        MtxDen& operator=(const MtxDen&)=delete;

public: void MultiplyMatrix(int K, double *X, double*Y);
public: void TransMultiplyMatrix(int K, double *X, double*Y);  //k is number of col of Y

public:
  int rows()const{return m_;}
  int cols()const{return n_;}

private:
  int m_,n_;
public:
  vector<double> a_;

public: virtual void myerror(const string s) {fprintf(stderr,"%s\n",s.c_str()); exit(1);}
public: virtual void dump(int indent=0);
};

/*
// ============================================================================
// MPI Matrix 
// ============================================================================
class MPIMtxDen: public MtxDen {
public: MPIMtxDen(const string& file_in, int ftn, int rank, int nproc);
public: MPIMtxDen(const string& file_in, int n, int k,  int rank, int nproc);
public: virtual ~MPIMtxDen(){};
public:
  MPIMtxDen(MPIMtxDen &&)=delete;
  MPIMtxDen(const MPIMtxDen &)=delete;
  MPIMtxDen& operator=(MPIMtxDen&&)=delete;
  MPIMtxDen& operator=(const MPIMtxDen&)=delete;

public:
  int get_nrows();
  int get_ncols();
  int get_nrows_loc();
  int get_ncols_loc();

private:
  int m_,n_;
  vector<double> a_;

public: virtual void myerror(const string s) {fprintf(stderr,"%s\n",s.c_str()); exit(1);}
public: virtual void dump(){};
};

*/



#endif

