/************************************************************************
  > File Name: src/mpi_mtx_basic.cpp
  > Author: xk
  > Descriptions: 
  > Created Time: Sun Dec 31 15:52:40 2017
 ************************************************************************/
#include "mtx_basic.hpp"

// ============================================================================
// constructor of a dense matrix 
// ----------------------------------------------------------------------------
// e.g. augmented sys E is a n by k dense matrix
// ============================================================================
MtxDen::MtxDen(const string& file_in){
  if(file_in.empty()) myerror("file is empty\n");
  FILE *fp = fopen(file_in.c_str(), "r");
  if(!fp) myerror("err: could not open fileE\n");
  MM_typecode matcode;
  mm_read_banner(fp, &matcode);
  if(   !mm_is_matrix(matcode)
      ||!mm_is_dense(matcode)
      ||!mm_is_array(matcode)
      ||!mm_is_general(matcode) ) {
    fclose(fp); fp=nullptr;
    myerror("for Init guess we only accept "
        "MatrixMarket matrix dense arrary general\n"
        "But the file is not\n");
  }
  double v;
  int m,n;
  mm_read_mtx_array_size(fp, &m, &n);
  m_=m; n_=n;
  a_.assign(m*n,.0);
  for(size_t j=0; j<n_; ++j){
    for(size_t i=0; i<m_; ++i){
      fscanf(fp, "%lg\n", &v);
      a_[i*n_+j]=v;  //read as row domian
      //e[i+j*m_]=v; //read as col domain
    }
  }
  fclose(fp);
  fp=nullptr;
}

// ============================================================================
// Y<- A*X       Y[ik]=A[ij]*X[jk]
// ----------------------------------------------------------------------------
// X is a row_domain, n_ by l matrix
// y is a row_domain, m_ by l matrix
// since row domain, uses for(ikj)
// ============================================================================
void MtxDen::MultiplyMatrix(int L, double *X, double *Y){  
  for(int i=0; i<m_; i++){
    const int itL = i*L;
    const int itN = i*n_;
    for(int k=0; k<L; k++) Y[itL+k]=0;
    for(int j=0; j<n_; j++){
      const int jtL = j*L;
      const double aij = a_[itN+j]; 
      for(int k=0; k<L; k++)
        Y[itL+k] += aij*X[jtL+k];
    }
  }
}

// ============================================================================
// Y<- A'*X       Y[ik]=A'[ij]*X[jk] = A[ji]*X[jk]
// ----------------------------------------------------------------------------
// A is a row_domain, m_ by n_ Matrix
// X is a row_domain, m_ by l matrix
// y is a row_domain, n_ by l matrix
// since row domain, uses for(ikj)
// ----------------------------------------------------------------------------
// input L should represent X's number of cols, and Y's number of cols
// ============================================================================
void MtxDen::TransMultiplyMatrix(int L, double *X, double *Y){
  
  //fprintf(stderr,"MtxDen m_%d,n_%d\n",m_,n_);
  
  for(int i=0; i<n_; i++){ //for i column of A, 
    const int itL = i*L;
    for(int k=0; k<L; k++) { Y[itL+k]=0;  
      //fprintf(stderr,"Y[i%d*L%d=itL%d+k%d]=0 ",i,L,itL,k);
    }
    //fprintf(stderr,"\n");
    for(int j=0; j<m_; j++){
      const int jtL = j*L;
      const int jtN = j*n_;
      const double aji = a_[jtN+i]; 
      //fprintf(stderr,"aji=a[j%d*N*%d=jtN%d+i]=%lg\n",j,n_,jtM, aji);
      for(int k=0; k<L; k++){
        Y[itL+k] += aji*X[jtL+k];
        //fprintf(stderr,"Y[%d]+=aji*X[%d] ",itL+k, jtL+k);
      }
      //fprintf(stderr,"\n");
    }
  }
}


// ============================================================================
// dump
// ============================================================================
void MtxDen::dump(int indent){
  fprintf(stderr,"\n\nMtxDen::dump()\n");
  for(int i=0;i<indent; i++) fprintf(stdout, " ");
  fprintf(stderr,"rows: %d, cols: %d\n",m_,n_);

  for(int i=0;i<indent; i++) fprintf(stdout, " ");
  fprintf(stderr,"a_(%d):",(signed)a_.size());
  for(auto x: a_) fprintf(stdout," %lg",x);
  fprintf(stderr,"\n\n");
}



