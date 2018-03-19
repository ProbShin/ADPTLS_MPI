/************************************************************************
  > File Name: src/mpi_mtx_basic.cpp
  > Author: xk
  > Descriptions: 
  > Created Time: Sun Dec 31 15:52:40 2017
 ************************************************************************/
#include "mtx_basic.hpp"


// ============================================================================
// constructor of sparse matrix using MPI
// ============================================================================
  MtxSpMPI::MtxSpMPI(const string& file_name, int rank, int nproc)
: MtxSp(),
  rank_(rank),
  nproc_(nproc)
{
  if(file_name.empty()) error("file name is empty()\n");
  FILE*fp = fopen(file_name.c_str(), "r");
  if(!fp) error(file_name," cannot be opened\n");
  MM_typecode matcode;
  mm_read_banner(fp, &matcode);
  if(   !mm_is_matrix(matcode)
      ||!mm_is_sparse(matcode)
      ||!mm_is_symmetric(matcode) ) {
    fclose(fp);
    fp=nullptr;
    error(file_name," MartixMarket format error, only accept\n"
        "MatrixMarket matrix coordinate real/int symmetric\n");
  }

  int r,c,nnz;
  mm_read_mtx_crd_size(fp, &r, &c, &nnz);

  num_rows_glb_ = r;
  num_cols_glb_ = c;
  num_cols_loc_ = c;

  double v;
  unordered_map<int, vector<pair<int, double>>> G;

  a_ .clear();
  ia_.clear();
  ja_.clear();

  for(int k=0; k<nnz; ++k) {
    fscanf(fp, "%d %d %lg\n", &r, &c, &v);
    if(r<c) { fclose(fp); fp=nullptr; error(file_name, "find nnz in upper triangular part.");} 
    r--; c--;  //base 0
    G[r].emplace_back(c,v);
    if(r!=c) G[c].emplace_back(r,v);
  }

  fclose(fp);
  fp =nullptr;

  for(auto it = G.begin(); it!=G.end(); ++it)
    sort((it->second).begin(), (it->second).end());

  // in case non-even split
  row_rcvcnt_.clear(); row_rcvcnt_.resize(nproc_, num_rows_glb_/nproc_);
  row_displs_.clear(); row_displs_.resize(nproc_+1, 0);

  for(int i=0, iEnd=num_rows_glb_%nproc_; i<iEnd; i++) row_rcvcnt_[i]++;
  for(int i=1; i<=nproc_; i++) row_displs_[i] = row_displs_[i-1] + row_rcvcnt_[i-1];
  
  num_rows_loc_ = row_rcvcnt_[rank_];

  // store G into CSR
  int rEnd;
  for(r=row_displs_[rank_],rEnd=row_displs_[rank_+1]; r!=rEnd; r++) {
    ia_.push_back((signed)a_.size());
    if(G.count(r)==0) continue;
    auto &Gr = G[r];
    for(auto&p : Gr) {
      ja_.push_back(p.first);
      a_ .push_back(p.second);
    }
  }
  ia_.push_back((signed)a_.size());
}

// ===========================================================================
// Construction function
// ===========================================================================
MtxSp::MtxSp(const string& file_name) {
  if(file_name.empty()) error("file name is empty()\n");
  FILE*fp = fopen(file_name.c_str(), "r");
  if(!fp) error(file_name," cannot be opened\n");
  MM_typecode matcode;
  mm_read_banner(fp, &matcode);
  if(   !mm_is_matrix(matcode)
      ||!mm_is_sparse(matcode)
      ||!mm_is_symmetric(matcode) ) {
    fclose(fp);
    fp=nullptr;
    error(file_name," MartixMarket format error, only accept\n"
        "MatrixMarket matrix coordinate real/int symmetric\n");
  }

  int r,c,nnz;
  mm_read_mtx_crd_size(fp, &r, &c, &nnz);

  num_rows_ = r;
  num_cols_ = c;

  double v;
  unordered_map<int, vector<pair<int, double>>> G;

  a_ .clear();
  ia_.clear();
  ja_.clear();

  for(int k=0; k<nnz; ++k) {
    fscanf(fp, "%d %d %lg\n", &r, &c, &v);
    if(r<c) { fclose(fp); fp=nullptr; error(file_name, "find nnz in upper triangular part.");} 
    r--; c--;  //base 0
    G[r].emplace_back(c,v);
    if(r!=c) G[c].emplace_back(r,v);
  }

  fclose(fp);
  fp =nullptr;

  for(auto it = G.begin(); it!=G.end(); ++it)
    sort((it->second).begin(), (it->second).end());
 
  for(r=0; r!=num_rows_; r++) {
    ia_.push_back((signed)a_.size());
    if(G.count(r)==0) continue;
    auto &Gr = G[r];
    for(auto&p : Gr) {
      ja_.push_back(p.first);
      a_ .push_back(p.second);
    }
  }
  ia_.push_back((signed)a_.size()); 
}


// ============================================================================
// back multiply a vector xglb into yglb.  Yglb <- MPI_Allgatherv( Aloc*xglb)
// ============================================================================
void MtxSpMPI::MultiplyVector_Allgatherv(int nrows_loc,        double *xglb, double *yloc, double*yglb, int*recvcounts, int*displs){
  MtxSp::MultiplyVector(nrows_loc,    xglb, yloc);
  MPI_Allgatherv(yloc, nrows_loc, MPI_DOUBLE, yglb, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD); 
} 


// ============================================================================
// back multiply a dense matrix locally, yloc <- allgatherv( Aloc*Xglb )
// ============================================================================
void MtxSpMPI::MultiplyMatrix_Allgatherv(int nrows_loc, int K, double *xglb, double* yloc, double*yglb, int*recvcounts, int*displs){
  MtxSp::MultiplyMatrix(nrows_loc, K, xglb, yloc);
  MPI_Allgatherv(yloc, nrows_loc*K, MPI_DOUBLE, yglb, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD); 
}


// ============================================================================
// back multiply a vector, Y=A*X
// ============================================================================
void MtxSp::MultiplyVector(int nrows, double*x, double* y){
  for(int i=0,iEnd=nrows; i<iEnd; i++){
    double t=0;
    for(int jt=ia_[i],jtEnd=ia_[i+1]; jt<jtEnd; jt++)
      t+=a_[jt]*x[ja_[jt]];
    y[i]=t;
  }
} //end of fun MultiplyVector

// ============================================================================
// back multiply a dense matrix, Y=A*X
// ----------------------------------------------------------------------------
// X is a nrows by K matrix
// Y[i][k] = A[i][j] * X[j][k]
// X is stored in row domain! for(i, k, j)
//   if X is not stored in row domain but col domain,
//   better to use for(k,i,j) or for(i,j,k)
// ============================================================================
void MtxSp::MultiplyMatrix(int nrows, int K, double*x, double* y){
#ifndef DENSE_MATRIX_COL_DOMAIN
  for(int i=0,iEnd=nrows; i<iEnd; i++){
    const int itK = i*K;
    for(int k=0; k<K; k++) y[itK+k]=0;
    for(int jt=ia_[i],jtEnd=ia_[i+1]; jt<jtEnd;jt++){
      const int j=ja_[jt];
      const int jtK = j*K;
      const double a=a_[jt];
      for(int k=0; k<K; k++) {
        y[itK+k]+=a*x[jtK+k];
      }
    }
  } 
#else
  for(int i=0,iEnd=nrows; i<iEnd; i++){
    const int itK = i*K;
    for(int k=0;k<K; k++){
      double t=0;
      for(int jt=ia_[i],jtEnd=ia_[i+1]; jt<jtEnd;jt++){
        t+=a_[jt]*xglb[ja_[jt]*K + k];
      }
      yloc[itK+k]=t;
    }
  }
#endif
} //end of fn MultiplyMatrix

// ============================================================================
// dump
// ============================================================================
void MtxSpMPI::dump(){
  MPI_Barrier(MPI_COMM_WORLD);
  int msg=0;
  if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  fprintf(stderr,"\n\n===========\nMtxSpMPI::dum() with rank %d nproc %d\n",rank_, nproc_);
  fprintf(stderr,"row_rcvcnt(%d) :",(signed)row_rcvcnt_.size());
  for(auto x: row_rcvcnt_) fprintf(stderr, " %d",x); fprintf(stderr,"\n");

  fprintf(stderr,"row_displs(%d) :",(signed)row_displs_.size());
  for(auto x: row_displs_) fprintf(stderr, " %d",x); fprintf(stderr,"\n");
  
  fprintf(stderr,"num_row_loc_ %d, num_col_loc_ %d, num_rows_glb_ %d, num_cols_glb_ %d\n",num_rows_loc_, num_cols_loc_, num_rows_glb_, num_cols_glb_);
  fprintf(stderr, "from Base class MtxSp:\n");
  fprintf(stderr," a_(%d) :",(signed) a_.size());
  for(int i=0;i<(signed)min((signed)a_.size(),100); i++) fprintf(stderr, " %g",a_[i]);
  if(a_.size()>100) fprintf(stderr, "...");
  fprintf(stderr,"\n");

  fprintf(stderr,"ia_(%d) :",(signed)ia_.size());
  for(int i=0;i<(signed)min((signed)ia_.size(),100); i++) fprintf(stderr, " %d",ia_[i]);
  if(ia_.size()>100) fprintf(stderr, "...");
  fprintf(stderr,"\n");
  
  fprintf(stderr,"ja_(%d) :",(signed)ja_.size());
  for(int i=0;i<(signed)min((signed)ja_.size(),100); i++) fprintf(stderr, " %d",ja_[i]);
  if(ja_.size()>100) fprintf(stderr, "...");
  fprintf(stderr,"\n");
  fprintf(stderr,"\n");

  if(rank_+1!=nproc_) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// error
// ============================================================================
void MtxSpMPI::error(const string&s1){
  fprintf(stderr, "Error in class MtxSpMPI, message\"%s\"\n",s1.c_str());
  exit(1);
}

// ============================================================================
// error
// ============================================================================
void MtxSpMPI::error(const string&s1, const string&s2){
  fprintf(stderr, "Error in class MtxSpMPI, message \"%s %s\"\n",s1.c_str(),s2.c_str());
  exit(1);
}



// ============================================================================
// dump
// ============================================================================
void MtxSp::dump(){
  fprintf(stderr, "\n\n==============\nMtxSp::dump()\n");
  fprintf(stderr, "num_rows_:%d, num_cols_:%d\n",num_rows_, num_cols_);
  fprintf(stderr, "a_(%d) :",(signed)a_.size());
  for(int i=0; i<(signed)min((signed)a_.size(),100); i++)
    fprintf(stderr, " %g",a_[i]);
  if(a_.size()>100) fprintf(stderr, "... \n");
  else fprintf(stderr,"\n");

  fprintf(stderr, "ia_(%d) :",(signed)ia_.size());
  for(int i=0; i<(signed)min((signed)ia_.size(),100); i++)
    fprintf(stderr, " %d",ia_[i]);
  if(ia_.size()>100) fprintf(stderr, "... \n");
  else fprintf(stderr,"\n");

  fprintf(stderr, "ja_(%d) :",(signed)ja_.size());
  for(int i=0; i<(signed)min((signed)ja_.size(),100); i++)
    fprintf(stderr, " %d",ja_[i]);
  if(ja_.size()>100) fprintf(stderr, "... \n");
  else fprintf(stderr,"\n");
}

// ============================================================================
// error
// ============================================================================
void MtxSp::error(const string&s1){
  fprintf(stderr, "Error in class MtxSP, message\"%s\"\n",s1.c_str());
  exit(1);
}

// ============================================================================
// error
// ============================================================================
void MtxSp::error(const string&s1, const string&s2){
  fprintf(stderr, "Error in class MtxSP, message \"%s %s\"\n",s1.c_str(),s2.c_str());
  exit(1);
}






