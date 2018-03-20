/*************************************************************************
    > File Name: src/mpi_mtx_ftol.cpp
    > Author: xk
    > Descriptions: 
    > Created Time: Sun Dec 31 16:11:02 2017
 ************************************************************************/

#include "def.hpp"
#include "ftmtx.hpp"
//#include <algorithm> //min
// ============================================================================
// constructor
// ============================================================================
FTMtxMPI::FTMtxMPI(const string&file_A, const string&file_E, int rank, int nproc, int ftn) :
 MtxSpMPI(rank,nproc)
{
  int &n    = G_sys_A_size_;
  int &k    = G_sys_E_size_; 
  vector<int>&nA_rcvcnt = row_rcvcnt_;
  vector<int>&nA_displs = row_displs_;
  vector<int>&nE_rcvcnt = G_nE_rcvcnt_;
  vector<int>&nE_displs = G_nE_displs_;


  MtxSpMPI *A = new MtxSpMPI(file_A, rank_, nproc_);
  MtxDen   *E = new MtxDen(file_E);
  
  if(A->rows() != E->rows()) error(file_A,"file_A rows != file_E rows ");
  // local sizes
  n = E->rows();
  k = E->cols();
  {
    auto& rcvcnt = A->get_row_rcvcnt(); 
    nA_rcvcnt.assign(rcvcnt.begin(), rcvcnt.end());
    auto& displs = A->get_row_displs();
    nA_displs.assign(displs.begin(), displs.end());
  }

  nE_rcvcnt.clear(); nE_rcvcnt.resize(nproc, k/nproc);
  for(int i=0, iEnd=k%nproc; i!=iEnd; i++) nE_rcvcnt[i]++;
  
  nE_displs.clear(); nE_displs.resize(nproc+1, 0);
  for(int i=1; i<=nproc; i++) nE_displs[i]=nE_displs[i-1]+nE_rcvcnt[i-1];
  
  num_rows_loc_ = nA_rcvcnt[rank];
  num_rows_glb_ = num_cols_glb_ = num_cols_loc_= n;

  const int nloc = num_rows_loc_;

  GAE.clear();   GAE.resize(nloc*k,0);
  GEA.clear();   GEA.resize(nE_rcvcnt[rank]*n,0);
  GEAE.clear();  GEAE.resize(nE_rcvcnt[rank]*k,0);

  double *AE = new double[n*k];
  double *EAE= new double[k*k];
  
  double *WorkSpaceAEloc = new double[nloc*k];
  vector<int> nMtx_rcvcnt(nA_rcvcnt.begin(), nA_rcvcnt.end());
  vector<int> nMtx_displs(nA_displs.begin(), nA_displs.end());
  for(auto &x : nMtx_rcvcnt) x*=k;
  for(auto &x : nMtx_displs) x*=k;
  A->MultiplyMatrix_Allgatherv(nloc, k, &(E->a_[0]), WorkSpaceAEloc, AE, &nMtx_rcvcnt[0], &nMtx_displs[0]);

  /*
#ifdef DEBUG_SHOW_DETAILS
  MPI_Barrier(MPI_COMM_WORLD);
  {
    int msg=0;
    if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  fprintf(stderr, "\n***\nFTMtxMPI::FTMtxMPI()\nn%d,k%d,nloc%d, rank%d,nproc%d\n",n,k,nloc, rank_, nproc_);
  
  fprintf(stderr, "nMtx_rcvcnt(%d): ",(signed)nMtx_rcvcnt.size());
  for(auto x: nMtx_rcvcnt) fprintf(stderr, " %d", x);
  fprintf(stderr,"\n");

  fprintf(stderr, "nMtx_displs(%d): ",(signed)nMtx_displs.size());
  for(auto x: nMtx_displs) fprintf(stderr, " %d", x);
  fprintf(stderr,"\n");

  fprintf(stderr, "AEloc<-Aloc*E\nAEloc[%d] :\n",nloc*k);
  for(int i=0; i<nloc; i++){
    for(int j=0; j<k; j++){
      fprintf(stderr, " %g",WorkSpaceAEloc[i*k+j]);
    }
    fprintf(stderr,"\n");
  }

  fprintf(stderr, "AE <- Allgather(AEloc)[%d] :\n",n*k);
  for(int i=0; i<n; i++){
    for(int j=0; j<k; j++){
      fprintf(stderr, " %g",AE[i*k+j]);
    }
    fprintf(stderr,"\n");
  }
  {
    int msg=0;
    if(rank_+1!=nproc) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  */

  delete []WorkSpaceAEloc; WorkSpaceAEloc=nullptr; 
  
  E->TransMultiplyMatrix(k, AE, EAE);

  /*
#ifdef DEBUG_SHOW_DETAILS
  MPI_Barrier(MPI_COMM_WORLD);
  {
    int msg=0;
    if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  fprintf(stderr, "EAE<-E'*AE; [%d] :\n",k*k);
  for(int i=0; i<k; i++){
    for(int j=0; j<k; j++){
      fprintf(stderr, " %g",EAE[i*k+j]);
    }
    fprintf(stderr,"\n");
  }
  {
    int msg=0;
    if(rank_+1!=nproc) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
*/
  delete E; E=nullptr;






  // G local 
  //unordered_map<int, unordered_map<int, double> > &G = G_;
  //{
  //  vector<double> &Aa  = A->get_a();
  //  vector<int> &Aia = A->get_ia();
  //  vector<int> &Aja = A->get_ja();
  //  for(int it=0, itEnd=nA_rcvcnt[rank_]; it<itEnd; it++) { //G += A
  //    int i = nA_displs[rank_]+it;
  //    for(int jt=Aia[it],jtEnd=Aia[it+1]; jt!=jtEnd; jt++){
  //      int    j = Aja[jt];
  //      double v = Aa [jt];
  //      G[i][j]=v;
  //    }
  //  }
  //}// prevent Aa, Aia and Aja been used further

  Gia.swap(A->get_ia());
  Gja.swap(A->get_ja());
  Ga .swap(A->get_a ());
  delete A;     A  =nullptr;
   
  for(int i=nA_displs[rank], iEnd=nA_displs[rank+1],iGAE=0; i<iEnd; i++,iGAE++) { //GAE += AE
    const int itk = i*k;
    for(int j=0; j<k; j++){
      double v = AE[ itk+j ];
      GAE[iGAE*k+j]=v;
    }
  }
  
  for(int i=0, iEnd=nE_rcvcnt[rank]; i<iEnd; i++) { //GEA += EA
    int iEA = rank+i*nproc;
    for(int j=0; j<n; j++){
      double v = AE[ j*k+iEA];
      GEA[i*n+j]=v;
    }
  }

  for(int i=0, iEnd=nE_rcvcnt[rank]; i<iEnd; i++) { //GEAE += EAE
    int iEAE = rank+i*nproc;
    for(int j=0; j<k; j++) {
      double v = EAE[ iEAE*k + j];
      GEAE[i*k+j]=v;
    }
  }
 
  //for(int i=nA_displs[rank], iEnd=nA_displs[rank+1]; i<iEnd; i++) { //G += AE
  //  for(int jt=0; jt<k; jt++){
  //   int    j = jt+n;
  //    double v = AE[ i*k+jt ];
  //    G[i][j]=v;
  //  }
  //}
 

  //for(int it=nE_displs[rank], itEnd=nE_displs[rank+1]; it<itEnd; it++) { //G+=EA
  //  int i = it+n;
  //  for(int j=0; j<n; j++) {
  //    double v = AE[ j*k+it ]; 
  //    G[i][j]=v;
  //  }
  //}
  
  //for(int it=nE_displs[rank], itEnd=nE_displs[rank+1]; it<itEnd; it++) { //G+=EAE
  //  int i = it+n;
  //  for(int jt=0; jt<k; jt++) {
  //    int    j = jt+n;
  //    double v = EAE[it*k+jt];
  //    G[i][j]=v; 
  //  }
  //}

  //for(int it=0, itEnd=nE_rcvcnt[rank]; it<itEnd; it++){ //G+=EA
  //  int iAE  = rank+it*nproc;
  //  int i    = iAE+n;
  //  for(int j=0; j<n; j++){
  //    double v = AE[ j*k + iAE];
  //    G[i][j]=v;
  //  }
  //}

  //for(int it=0, itEnd=nE_rcvcnt[rank]; it<itEnd; it++) { //G+=EAE
  //  int iEAE  = rank+it*nproc;
  //  int i     = iEAE + n;
  //  for(int jt=0; jt<k; jt++){
  //    int    j= jt+n;
  //    double v= EAE[iEAE*k + jt];
  //    G[i][j]=v;
  //  }
  //}
  


  delete []AE; AE =nullptr;
  delete []EAE;EAE=nullptr;

/*  
#ifdef DEBUG_SHOW_DETAILS
  MPI_Barrier(MPI_COMM_WORLD);
  {
    int msg=0;
    if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  fprintf(stderr,"rank%d tring to disply G_loc\n",rank_);
  int cnt=0;
  vector<int> candi;
  for(auto p=G_.begin(), pEnd=G_.end(); p!=pEnd; p++)
    candi.push_back(p->first);
  sort(candi.begin(), candi.end());
  for(auto i : candi){
    if(cnt++>=100) {fprintf(stderr,"...\n"); break; }
    fprintf(stderr, "G[%d]:",i);
    auto &L = G_[i];
    vector<int> candnb;
    for(auto p=L.begin(), pEnd=L.end(); p!=pEnd; p++){
      candnb.push_back(p->first);
    }
    sort(candnb.begin(), candnb.end());
    int nbcnt=0;
    for(auto nb : candnb){
      if (nbcnt++>=100) {fprintf(stderr,"...\n"); break; }
      fprintf(stderr,"%g(%d)\t",L[nb], nb);
    }
    fprintf(stderr, "\n");
  } 
  
  {
    int msg=0;
    if(rank_+1!=nproc) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif 
*/




  // G(rowIndex, colIndex) stored into csr format
  fault_boom(ftn);
} // end of fn FTMtxMPI()


// ============================================================================
// fault happen
// ============================================================================
void FTMtxMPI::fault_boom(int ftn){
  
  vector<int> &ia = MtxSp::get_ia();
  vector<int> &ja = MtxSp::get_ja();
  vector<double> & a = MtxSp::get_a ();
  
  a.clear();
  ia.clear();
  ja.clear();
 
  const int rank = rank_;
  const int nproc = nproc_;

  const int n = G_sys_A_size_;
  const int k = G_sys_E_size_;

  if(ftn>k) {fprintf(stderr,"Error FTMtxMPI::fault_boom() tries to genereate faults %d > faulted system tolerant size k %d.\n",ftn,k); exit(1);}
  vector<int>&nA_rcvcnt = row_rcvcnt_;
  vector<int>&nA_displs = row_displs_;
  vector<int>&nE_rcvcnt = G_nE_rcvcnt_;
  vector<int>&nE_displs = G_nE_displs_;

  vector<int> nF_rcvcnt;
  nF_rcvcnt.clear();
  nF_rcvcnt.resize(nproc, ftn/nproc);
  for(int i=0, iEnd=ftn%nproc; i!=iEnd; i++) nF_rcvcnt[i]++;

  //unordered_map<int, unordered_map<int, double> > &G = G_;
  
  // get row_indexs, and col_indexs
  vector<int> row_indexs,col_indexs;
    // rows
  for(int i=0, iEnd=nF_rcvcnt[rank]; i<iEnd; i++){
    row_indexs.push_back(rank+i*nproc+n);   //faulted_rows replaced by augmented rows
  }
  for(int i=nA_displs[rank]+nF_rcvcnt[rank], iEnd=nA_displs[rank+1]; i<iEnd; i++) {
    row_indexs.push_back(i);                //remained rows
  }
    // cols
  for(int pid=0; pid<nproc; pid++) {
    for(int i=0, iEnd=nF_rcvcnt[pid];i<iEnd;i++) {
      col_indexs.push_back(pid+i*nproc+n);
    }
    for(int i=nA_displs[pid]+nF_rcvcnt[pid], iEnd=nA_displs[pid+1]; i<iEnd; i++) {
      col_indexs.push_back(i);
    }
  }

#ifdef DEBUG_SHOW_DETAILS
  MPI_Barrier(MPI_COMM_WORLD);
  {
    int msg=0;
    if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  fprintf(stderr,"rank%d / nproc%d fault boom() \n",rank_, nproc_);
  
  
  
  
  fprintf(stderr,"row_indexs(%d) :",(signed)row_indexs.size());
  for(int i=0; i<(signed)min((signed)row_indexs.size(),100); i++)
    fprintf(stderr," %d",row_indexs[i]);
  if(row_indexs.size()>100) fprintf(stderr,"...");
  fprintf(stderr,"\n");

  fprintf(stderr,"col_indexs(%d) :",(signed)col_indexs.size());
  for(int i=0; i<(signed)min((signed)col_indexs.size(),100); i++)
    fprintf(stderr," %d",col_indexs[i]);
  if(col_indexs.size()>100) fprintf(stderr,"...");
  fprintf(stderr,"\n");
  
  {
    int msg=0;
    if(rank_+1!=nproc) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif 



  ia.clear(); ia.reserve(n+1);
  ja.clear(); ja.reserve(Gja.size());
  a.clear();  a.reserve(Ga.size());

  const int RBeg = nA_displs[rank];
  for(auto r: row_indexs){
    ia.push_back((signed)ja.size());
    if(r<n){
      for(int j=0; j<n; j++){
        int c = col_indexs[j];
        if(c<n){
          for(int jt=Gia[r-RBeg],jtEnd=Gia[r-RBeg+1]; jt<jtEnd; jt++) {
            if(Gja[jt]==c) {
              ja.push_back(j);
               a.push_back(Ga[jt]);
               break;
            }
          }
        }
        else{
          ja.push_back(j);
          a .push_back(GAE[(r-RBeg)*k+c-n]);
        }
      }
    }
    else{
      for(int j=0,jEnd=n; j<jEnd; j++) {
        ja.push_back(j);
        int c = col_indexs[j];
        if(c<n)
          a.push_back(GEA[(r-n)/nproc*n+c]);
        else
          a.push_back(GEAE[(r-n)/nproc*k+c-n]);
      }   
    }
  }
  ia.push_back((signed)ja.size());

  /*
  unordered_map<int, unordered_map<int, double> >::const_iterator GEnd = G.end();
  unordered_map<int, unordered_map<int, double> >::const_iterator Gr;
  const int RBEG = nA_displs[rank];
  ia.clear(); ia.reserve(n+1);
  ja.clear();
  for(auto r:row_indexs){
    ia.push_back((signed)ja.size());
    if(r<n){
      auto Gr = G.find(r);
      for(int j=0,jEnd=n; j<jEnd; j++){
        int c = col_indexs[j];
        if(c<n){
          if(Gr==GEnd) continue;
          auto Grc = Gr->second.find(c);
          if(Grc==Gr->second.end()) continue;
          ja.push_back(j);
          a .push_back(Grc->second);
        }
        else{
          ja.push_back(j);
          a .push_back(GAE[(r-RBEG)*k+c-n]);
        }
      } // endof for j in n
    } // endof r<n
    else{
      for(int j=0,jEnd=n; j<jEnd; j++) {
        ja.push_back(j);
        int c = col_indexs[j];
        if(c<n)
          a.push_back(GEA[(r-n)/nproc*n+c]);
        else
          a.push_back(GEAE[(r-n)/nproc*k+c-n]);
      }
    }
  }
  ia.push_back((signed)ja.size());
*/

/*
  // G(row_indexs, col_indexs) to CSR
  unordered_map<int, unordered_map<int, double> >::const_iterator GEnd = G.end();
  unordered_map<int, unordered_map<int, double> >::const_iterator Gr;
  
  unordered_map<int, double>::const_iterator GrEnd;
  unordered_map<int, double>::const_iterator Grc;
  for(auto r : row_indexs){
    ia.push_back((signed)ja.size());
    if((Gr=G.find(r)) == GEnd) continue;
    auto &Grmap = Gr->second;
    GrEnd = Grmap.end();
    for(int c=0; c<n; c++) {
      int act_cols = col_indexs[c];
      if((Grc=Grmap.find(act_cols)) == GrEnd) continue;
      ja.push_back(c);
      a .push_back(Grc->second);
    }
  }
  ia.push_back((signed)ja.size());
*/
} //end of fault_boom

// ============================================================================
// Atut(deleted_rows, replaced/remained_cols) scale per row, and sum by row
// ============================================================================
void FTMtxMPI::Atut_sub_mtx_times_x_saved(vector<int> &rows, vector<int>&cols, vector<double>&args, double*out){
/*  const int n = (signed)cols.size();
  for(int i=0; i<n; i++) out[i] = 0;
  for(int i=0, iEnd=(signed)rows.size(); i<iEnd; i++){
    int r = rows[i];
    double a = args[i];
    auto Git = G_.find(r);
    if(Git==G_.end()) continue;
    auto &GG = Git->second;

    for(int j=0; j<n; j++){
      int c = cols[j];
      auto GGit = GG.find(c);
      if(GGit==GG.end()) continue;
      out[j]+=a*GGit->second;
    }
  }
  */

  /*
  const int n = G_sys_A_size_;
  const int k = G_sys_E_size_;        
  const int RBeg     = row_displs_[rank_];
  const int nCols = (signed)cols.size();
  const int nRows = (signed)rows.size();

  for(int i=0; i<nCols; i++) out[i]=0;
  for(int i=0; i<nRows; i++){
    int r = rows[i];
    double a = args[i];
    if(r<n){
      auto Gr = G_.find(r);
      for(int j=0; j<nCols; j++){
        int c = cols[j];
        if(c<n){
          if(Gr == G_.end()) continue;
          auto Grc = Gr->second.find(c);
          if(Grc == Gr->second.end()) continue;
          out[j]+=a*Grc->second;
        }
        else{
          out[j]+=a*GAE[(r-RBeg)*k+c-n];
        }
      }
    }
    else{
      for(int j=0; j<nCols; j++){
        int c = cols[j];
        if(c<n) out[j]+=a*GEA[ (r-n)/nproc_*n+c];
        else    out[j]+=a*GEAE[(r-n)/nproc_*n+c-n];
      }
    }
  }
  */

  const int n = G_sys_A_size_;
  const int k = G_sys_E_size_;        
  const int RBeg     = row_displs_[rank_];
  const int nCols = (signed)cols.size();
  const int nRows = (signed)rows.size();

  for(int i=0; i<nCols; i++) out[i]=0;
  for(int i=0; i<nRows; i++){
    int r = rows[i];
    double a = args[i];
    if(r<n){
      for(int j=0; j<nCols; j++){
        int c = cols[j];
        if(c<n){
          for(int jt=Gia[r-RBeg], jtEnd=Gia[r-RBeg+1]; jt<jtEnd; jt++){
            if(Gja[jt]==c){
              out[j]+=a*Ga[jt];
              break;
            }
          }
        }
        else{
          out[j]+=a*GAE[(r-RBeg)*k+c-n];
        }
      }
    }
    else{
      for(int j=0; j<nCols; j++){
        int c = cols[j];
        if(c<n) out[j]+=a*GEA[ (r-n)/nproc_*n+c];
        else    out[j]+=a*GEAE[(r-n)/nproc_*n+c-n];
      }
    }
  }




}


// ============================================================================
// dump
// ============================================================================
void FTMtxMPI::dump(){
  MPI_Barrier(MPI_COMM_WORLD);
  int msg=0;
  if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  fprintf(stderr,"\n\n==========\nFTMtxMPI::dump() with rank%d nproc%d\n",rank_, nproc_);
  fprintf(stderr, "n which named as G_sys_A_size = %d, k which named as G_sys_E_size = %d\n",G_sys_A_size_, G_sys_E_size_);
  int cnt=0;


  fprintf(stderr, "Gia (%d) :",(signed)Gia.size());
  for(auto x: Gia) fprintf(stderr," %d",x); fprintf(stderr,"\n");

  fprintf(stderr, "Gja (%d) :",(signed)Gja.size());
  for(auto x: Gja) fprintf(stderr," %d",x); fprintf(stderr,"\n");

  fprintf(stderr, "Ga  (%d) :",(signed)Ga.size());
  for(auto x: Ga) fprintf(stderr," %g",x); fprintf(stderr,"\n");
  /*
  vector<int> candi;
  for(auto p=G_.begin(), pEnd=G_.end(); p!=pEnd; p++)
    candi.push_back(p->first);
  sort(candi.begin(), candi.end());
  for(auto i : candi){
    if(cnt++>=100) {fprintf(stderr,"...\n"); break; }
    fprintf(stderr, "G[%d]:",i);
    auto &L = G_[i];
    vector<int> candnb;
    for(auto p=L.begin(), pEnd=L.end(); p!=pEnd; p++){
      candnb.push_back(p->first);
    }
    sort(candnb.begin(), candnb.end());
    int nbcnt=0;
    for(auto nb : candnb){
      if (nbcnt++>=100) {fprintf(stderr,"...\n"); break; }
      fprintf(stderr,"%g(%d)\t",L[nb], nb);
    }
    fprintf(stderr, "\n");
  }
  */

/*
  auto p=G_.begin();
  for(; p!=G_.end() && cnt++<100; p++){
    fprintf(stderr,"G[%d] :",p->first);
    int nbcnt=0;
    auto pp=p->second.begin();
    for(auto ppEnd=p->second.end(); pp!=ppEnd && nbcnt++<100; pp++){
      fprintf(stderr," %d(%g)",pp->first, pp->second);
    }
    if(p->second.size()>100) fprintf(stderr,"...");
    fprintf(stderr, "\n");
  }
  if(G_.size()>100) fprintf(stderr, "...");
  fprintf(stderr,"\n");
*/
  fprintf(stderr, "GAE:\n");
  cnt=0; for(auto x: GAE) { fprintf(stderr," %g",x); if(++cnt%G_sys_E_size_==0) fprintf(stderr,"\n"); }
  fprintf(stderr, "GEA:\n");
  cnt=0; for(auto x: GEA) { fprintf(stderr," %g",x); if(++cnt%G_sys_A_size_==0) fprintf(stderr,"\n"); }
  fprintf(stderr, "GEAE:\n");
  cnt=0; for(auto x: GEAE){ fprintf(stderr," %g",x); if(++cnt%G_sys_E_size_==0) fprintf(stderr,"\n"); }
  fprintf(stderr,"\nfrom base's FTMtxMPI::MtxSpMPI::\n");
  fprintf(stderr,"row_rcvcnt(%d) :",(signed)row_rcvcnt_.size());
  for(auto x: row_rcvcnt_) fprintf(stderr, " %d",x); fprintf(stderr,"\n");

  fprintf(stderr,"row_displs(%d) :",(signed)row_displs_.size());
  for(auto x: row_displs_) fprintf(stderr, " %d",x); fprintf(stderr,"\n");


  fprintf(stderr,"\nfrom base's base's FTMtxMPI::MtxSpMPI::MtxSp::\n");
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


  if(rank_+1!=nproc_) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// error
// ============================================================================
void FTMtxMPI::error(const string&s1) {
  fprintf(stderr, "Error in class FTMtxMPI with message \"%s\"\n",s1.c_str());
  exit(1);
}


// ============================================================================
// error
// ============================================================================
void FTMtxMPI::error(const string&s1, const string&s2) {
  fprintf(stderr, "Error in class FTMtxMPI with message \"%s %s\"\n",s1.c_str(), s2.c_str());
  exit(1);
}


