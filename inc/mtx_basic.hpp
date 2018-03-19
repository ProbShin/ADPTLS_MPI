#ifndef MPI_MTX_BASIC_HPP
#define MPI_MTX_BASIC_HPP

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

  void MultiplyVector(int nrows, double* x, double *y);             
  void MultiplyMatrix(int nrows, int ncols, double* x, double *y);     


public:virtual void dump();
private:
virtual void error(const string& s1);
virtual void error(const string& s1, const string&s2);

protected:
vector<int> ia_;
vector<int> ja_;
vector<double> a_;

private:
int num_rows_;
int num_cols_;
};

// ============================================================================
// MPI Distributed Matrix
// ============================================================================
class MtxSpMPI : public MtxSp {
public: MtxSpMPI(int rnk, int np):MtxSp(),rank_(rnk), nproc_(np) {}
public: MtxSpMPI(const string&file_name, int rnk, int np); 
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
  void MultiplyVector_Allgatherv(int nrows,        double* xglb, double *yloc, double *yglb, int* recvcounts, int* displs);
  void MultiplyMatrix_Allgatherv(int nrows, int K, double* xglb, double *yloc, double *yglb, int* recvcounts, int* displs);

public:
  virtual int rows() const { return num_rows_glb_; }
  virtual int cols() const { return num_cols_glb_; }
  virtual int rows_loc() const { return num_rows_loc_; }
  virtual int cols_loc() const { return num_cols_loc_; }
  
  vector<int>& get_row_displs(){ return row_displs_; }
  vector<int>& get_row_rcvcnt(){ return row_rcvcnt_; }

public:  virtual void dump();
private: virtual void error(const string& s1);
private: virtual void error(const string& s1,const string& s2);

protected:
const int rank_;
const int nproc_;

vector<int> row_rcvcnt_;
vector<int> row_displs_;

protected:
int num_rows_loc_, num_cols_loc_;
int num_rows_glb_, num_cols_glb_;

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

