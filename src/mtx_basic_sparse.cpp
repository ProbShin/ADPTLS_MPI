/************************************************************************
  > File Name: src/mpi_mtx_basic.cpp
  > Author: xk
  > Descriptions: 
  > Created Time: Sun Dec 31 15:52:40 2017
 ************************************************************************/
#include "mtx_basic.hpp"
#include <unordered_set>
#include <fstream>
// ============================================================================
// constructor of sparse matrix using MPI
// ============================================================================
  MtxSpMPI::MtxSpMPI(const string& file_name, int rank, int nproc)
: MtxSp(),
  rank_(rank),
  nproc_(nproc)
{

  a_ .clear();
  ia_.clear();
  ja_.clear();
  
  int r,c;
  if(file_name.size()<4 || file_name.substr(file_name.size()-4, 4)!=".mtx"){
    do_read_binary_mtx_into_CSR(file_name, 
        num_rows_glb_, num_cols_glb_, num_rows_loc_, num_cols_loc_, 
        row_rcvcnt_, row_displs_,
        rank_, nproc_);
  }
  else{

    unordered_map<int, vector<pair<int, double>>> G;
    do_read_mm_mtx_into_G(file_name, G, r, c);
    num_rows_glb_ = r;
    num_cols_glb_ = c;
    num_cols_loc_ = c;
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
 

  
  // prepare for matvec mpi fun, 
  mv_x_sndcnt_.clear(); mv_x_sndcnt_.resize(nproc_  , 0);
  mv_x_sndpls_.clear(); mv_x_sndpls_.resize(nproc_+1, 0);
  mv_x_rcvcnt_.clear(); mv_x_rcvcnt_.resize(nproc_  , 0);
  mv_x_rcvpls_.clear(); mv_x_rcvpls_.resize(nproc_+1, 0);


  // get unique non zero cols of A_{rank,pid}  
  int nloc = row_rcvcnt_[rank_];
  vector<unordered_set<int>> v_uniq_rcs(nproc_);
  for(int it=0; it<nloc; it++) {
    int curPidPlus1 =1;
    for(int jt=ia_[it], jtEnd=ia_[it+1]; jt!=jtEnd; jt++) {
      int    col = ja_[jt]; //printf("find col %d\n",col);
      double val =  a_[jt];
      while( row_displs_[curPidPlus1]<= col) curPidPlus1++;
      v_uniq_rcs[curPidPlus1-1].insert(col);
    }
  }
  
  // set up recive_buffer, recive_count, recive_displs, and recive_indexs
  for(int pid=0; pid<nproc_; pid++){
    mv_x_rcvcnt_[pid]  =v_uniq_rcs[pid].size();
    mv_x_rcvpls_[pid+1]=mv_x_rcvpls_[pid]+mv_x_rcvcnt_[pid];

  }
  int rcvbuf_size = mv_x_rcvpls_[nproc_];
  mv_x_rcvbuf_.clear(); mv_x_rcvbuf_.resize(rcvbuf_size,0);
  mv_x_rcvidx_.reserve(rcvbuf_size);
  for(int pid=0; pid<nproc_; pid++){
    vector<int> idxs(v_uniq_rcs[pid].begin(), v_uniq_rcs[pid].end()); //Required Cols from Process pId
    sort(idxs.begin(), idxs.end());
    mv_x_rcvidx_.insert(mv_x_rcvidx_.end(), idxs.begin(), idxs.end());
  }

  // set up sender_buffer, sender_count, sender_displs and sender_indexs
  MPI_Alltoall(&mv_x_rcvcnt_[0], 1, MPI_INT, &mv_x_sndcnt_[0], 1, MPI_INT, MPI_COMM_WORLD);
  for(int i=0; i<nproc; i++) mv_x_sndpls_[i+1]=mv_x_sndpls_[i]+mv_x_sndcnt_[i];
  int sndbuff_size = mv_x_sndpls_[nproc_];

  mv_x_sndidx_.clear(); mv_x_sndidx_.resize(sndbuff_size,0);
  mv_x_sndbuf_.clear(); mv_x_sndbuf_.resize(sndbuff_size,0);
  MPI_Alltoallv(&mv_x_rcvidx_[0], &mv_x_rcvcnt_[0], &mv_x_rcvpls_[0], MPI_INT, 
      &mv_x_sndidx_[0], &mv_x_sndcnt_[0], &mv_x_sndpls_[0], MPI_INT, MPI_COMM_WORLD);
  
  // snder_indexs <- global_to_local(snder indexs) 
  int disp = row_displs_[rank_];
  for(int i=0, iEnd=mv_x_sndidx_.size(); i<iEnd; i++) 
    mv_x_sndidx_[i]-=disp;


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
// back multiply a vector        Y_{nloc,1} <- A_{nloc, N}*x_{nloc,1}
// ----------------------------------------------------------------------------
// input: nrows_loc should be nloc
// input: xloc is array(bloc by 1), contains xloc data
// input: xglob_workspace is array(n by 1), does not needed to be init
// output: yloc is array(n by 1)
// ----------------------------------------------------------------------------
// procedure:
//    pack the needed information
//    communicate the needed information
//    unpack the needed informaiton
//    MatVec operation
// ============================================================================
void MtxSpMPI::MultiplyVectorMPI(int nrows_loc, double *xloc, double *xglb_workspace, double*yloc){
  // pack for the sender buffer
  for(int i=0, iEnd=mv_x_sndidx_.size(); i<iEnd; i++)
    mv_x_sndbuf_[i]=xloc[ mv_x_sndidx_[i] ];

  MPI_Alltoallv(&mv_x_sndbuf_[0], &mv_x_sndcnt_[0], &mv_x_sndpls_[0], MPI_DOUBLE, 
      &mv_x_rcvbuf_[0], &mv_x_rcvcnt_[0], &mv_x_rcvpls_[0], MPI_DOUBLE, MPI_COMM_WORLD);

  // unpack recved buffer
  for(int i=0, iEnd=mv_x_rcvbuf_.size(); i<iEnd; i++) 
    xglb_workspace[mv_x_rcvidx_[i]] = mv_x_rcvbuf_[i];
  
  MtxSp::MultiplyVector(nrows_loc, xglb_workspace, yloc);
}


// ============================================================================
// back multiply a vector xglb into yglb.  Yglb <- MPI_Allgatherv( Aloc*xglb)
// ============================================================================
//void MtxSpMPI::MultiplyVector_Allgatherv(int nrows_loc,        double *xglb, double *yloc, double*yglb, int*recvcounts, int*displs){
//  MtxSp::MultiplyVector(nrows_loc,    xglb, yloc);
//  MPI_Allgatherv(yloc, nrows_loc, MPI_DOUBLE, yglb, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD); 
//} 


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
//
// ============================================================================
void MtxSp::TransMultiplyVector(int nrows, double*x, double* y ){
  for(int i=0,iEnd=cols(); i<iEnd; i++) y[i]=0;
  for(int i=0,iEnd=nrows;  i<iEnd; i++){
    double alpha = x[i]; 
    if(alpha==0) continue;
    for(int jt=ia_[i],jtEnd=ia_[i+1]; jt<jtEnd; jt++)
      y[ja_[jt]]+=alpha*a_[jt];
  }
} //end of fun MultiplyVector


// ============================================================================
// fuck
// ============================================================================
void MtxSpMPI::do_read_mm_mtx_into_G(const string& file_name, unordered_map<int,vector<pair<int,double>>> &G, int& nrows, int&ncols){
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

  int r,c, nnz;
  mm_read_mtx_crd_size(fp, &r, &c, &nnz);
  nrows=r; 
  ncols=c;
  double v;

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
} //end of function do_read_mm_mtx_into_G

// ============================================================================
// fuck
// ============================================================================
void MtxSpMPI::do_read_binary_mtx_into_CSR(const string& file_name, int& rglb, int&cglb, 
    int&rloc, int &cloc, 
    vector<int> &row_rcvcnt, vector<int> &row_displs, 
    const int rank, const int nproc){
  if(file_name.empty()) error("file name is empty()\n");
  ifstream myFile (file_name.c_str(), ios::in | ios::binary);

  if(sizeof(long int)!=8) { printf("Err! this machine 'long int' is not 64bit. Cannot read file %s\n",file_name.c_str()); exit(1); }
  long int nrows64, ncols64, nnz;
  myFile.read((char*)&nrows64, sizeof(long int));
  myFile.read((char*)&ncols64, sizeof(long int));
  myFile.read((char*)&nnz,     sizeof(long int));
 
  vector<long int> colptr(ncols64+1);
  vector<long int> rowval(nnz);
  vector<double>   nzval (nnz);
  
  myFile.read((char*) &colptr[0], (ncols64+1)*sizeof(long int));
  myFile.read((char*) &rowval[0], nnz    *sizeof(long int));
  myFile.read((char*) &nzval [0], nnz    *sizeof(long int));

  myFile.close();

    rglb = nrows64;
    cglb = ncols64;
    cloc = ncols64;
    
    row_rcvcnt.clear(); row_rcvcnt.resize(nproc, rglb/nproc);
    row_displs.clear(); row_displs.resize(nproc+1, 0);

    for(int i=0, iEnd=rglb%nproc; i<iEnd; i++) row_rcvcnt[i]++;
    for(int i=1; i<=nproc; i++) row_displs[i] = row_displs[i-1] + row_rcvcnt[i-1];
    rloc = row_rcvcnt[rank];
    

    ia_.clear();
    ja_.clear();
    a_.clear();

    for(int r=row_displs[rank], rEnd=row_displs[rank+1]; r!=rEnd; r++){
      ia_.push_back((signed)a_.size());
      for(int jt=colptr[r],jtEnd=colptr[r+1]; jt!=jtEnd; jt++){
        ja_.push_back(rowval[jt]);
         a_.push_back(nzval [jt]);
      }
    }
    ia_.push_back((signed)a_.size());

} //end of function do_read_mm_mtx_into_G




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






