/*************************************************************************
    > File Name: src/mpi_adls.cpp
    > Author: xk
    > Descriptions: 
    > Created Time: Sun Dec 31 17:00:01 2017
 ************************************************************************/
#include "def.hpp"
#include "adptls.hpp"
#include <sstream>





// ============================================================================
// Liner Solver Fault system
// ============================================================================
LSFaultMPI::LSFaultMPI(const string&fA, const string&fE, const string& fb, 
    vector<int>& vfts, const string& ftfa_base, 
    int rk, const int np){
  ftn=0;
  rank=rk;
  nproc=np;
  A4Res = new MtxSpMPI(fA, "mm", rank, nproc);
  Atut  = new FTMtxMPI(fA, fE, rank, nproc, 0);
  E4Res = new MtxDen(fE);
  b4Res = new MtxDen(fb);
  Eb=nullptr;  //for faulted rhs, b<- E'b
  {
    MtxDen E(fE);
    Eb = new double[E.cols()];
    E.TransMultiplyMatrix(1, &(b4Res->a_[0]), Eb);
  }

  rstep_pool.clear();
  for(int i=vfts.size()-1; i>=0; i--) {
    rstep_pool.emplace_back(vfts[i]);
  }
  stringstream ss;
  for(int i=vfts.size(); i>0; i--) {
    ss<<ftfa_base<<"_K"<<Atut->get_sys_E_size()<<"_"<<nproc<<"p"<<"_s"<<i<<".mtx";
    rfA_pool.push_back(ss.str());
    ss.str("");
  }

  // fault recover variables

} //end of construction



void LSFaultMPI::boom(MtxSpMPI* &A, double* array_nby1, double* x_loc, vector<double>&saved_x_loc, double* bloc, double*r_loc, double& rddot_loc, double& rddot, double&beta) {
      ftn += 1;
      if(!rstep_pool.empty()) rstep_pool.pop_back();

  int ftn_loc=0;
  int ftn_pre_loc=0;
  vector<int> ftn_rcvcnt;
      ftn_rcvcnt.clear();
      ftn_rcvcnt.resize(nproc, ftn/nproc);
      for(int i=0, iEnd=ftn%nproc; i!=iEnd; i++) ftn_rcvcnt[i]++;
      ftn_pre_loc = ftn_loc;
      ftn_loc = ftn_rcvcnt[rank];

      // update A
      delete A; 
      A =new MtxSpMPI(rfA_pool.back().c_str(), "mm", rank, nproc);
      rfA_pool.pop_back();
      
      int nloc = A->rows_loc();  
      int n    = A->rows();

      int *row_rcvcnt = &((A->get_row_rcvcnt())[0]); //A_rcvcnt[0];
      int *row_displs = &((A->get_row_displs())[0]); //A_displs[0];
      
      
      // update x  
      if(ftn_pre_loc!=ftn_loc){
        saved_x_loc.push_back(x_loc[ftn_pre_loc]);
        x_loc[ftn_pre_loc]=0;
      }
  
      // update right hand side.  b<- b.update(E'b) - Atut(fault_related row,cols) * saved_x 
      double* orig_rhs_loc = &b4Res->a_[row_displs[rank]];
      for(int i=0; i<nloc; i++)
        bloc[i] = orig_rhs_loc[i]; // = b4Res->a_[row_displs[rank]];
      for(int i=0; i<ftn_loc; i++) {
        bloc[i] = Eb[rank+i*nproc];
      }

      double *ft_b_per_rank = new double[n];
      vector<int> ft_related_rows, ft_related_cols;
      
      for(int i=0,iEnd=ftn_rcvcnt[rank]; i!=iEnd; i++){
        ft_related_rows.push_back(row_displs[rank]+i);  //faulted rows
      }
      
      for(int pid=0; pid!=nproc; pid++){
        for(int i=0; i<ftn_rcvcnt[pid]; i++){
          ft_related_cols.push_back(pid+i*nproc+n);
        }
        for(int i=row_displs[pid]+ftn_rcvcnt[pid], iEnd=row_displs[pid+1]; i<iEnd; i++){
          ft_related_cols.push_back(i);
        }
      }

      Atut->Atut_sub_mtx_times_x_saved(ft_related_rows, ft_related_cols, saved_x_loc, ft_b_per_rank); 

      double *ft_b          = new double[n];
      MPI_Allreduce(ft_b_per_rank, ft_b, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      delete []ft_b_per_rank; ft_b_per_rank=nullptr;
      double *ft_b_loc = &ft_b[row_displs[rank]];
      for(int i=0; i<nloc; i++)
        bloc[i]-=ft_b_loc[i];

      delete []ft_b; ft_b=nullptr;
      ft_b_loc = nullptr;

      // update r_loc = b-Ax
      A->MultiplyVectorMPI(nloc, &x_loc[0], array_nby1, &r_loc[0]);
      //MPI_Allgatherv(x_loc, nloc, MPI_DOUBLE, &v_x_[0], row_rcvcnt, row_displs, MPI_DOUBLE, MPI_COMM_WORLD);
      //A->MultiplyVector(nloc, &(v_x_[0]), r_loc);
      for(int i=0; i<nloc;i++) r_loc[i]=bloc[i] - r_loc[i];

      //update rddot_loc and rddot
      rddot_loc=.0; for(int i=0; i<nloc; i++) rddot_loc+=r_loc[i]*r_loc[i];
      MPI_Allreduce(&rddot_loc,&rddot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      beta=.0;

}







// ----------------------------------------------------------------------------
// MPI versioned Adaptive Linear Solver, (Conjugate Gradient)
// ----------------------------------------------------------------------------
void MPI_ADLS_CG::solve(){

  const int &rank  = rank_;
  const int &nproc = nproc_;   
  
  // CG variables
  MtxSpMPI* &A     = A_;      //A
  MtxDen    &b     = b_;      //rhs
  int nloc = A->rows_loc();  
  int n    = A->rows();
  int *row_rcvcnt = &((A->get_row_rcvcnt())[0]); //A_rcvcnt[0];
  int *row_displs = &((A->get_row_displs())[0]); //A_displs[0];
  int disp  = row_displs[rank];
  double *p_loc   = &(v_p_loc_[0]);
  //double *p       = &(v_p_[0]);
  double *Ap_loc  = &(v_Ap_loc_[0]);
  double *x_loc   = &(v_x_loc_[0]);
  double *r_loc   = &(v_r_loc_[0]);
  double rddot_loc=.0;
  double curr_rddot=.0;
  double prev_rddot=.0;
  double pAp_loc=0;
  double pAp =0;
  double *rhs_loc = &(b.a_[disp]);
  vector<double> mem_space_nby1(n);
  double *array_nby1 = &mem_space_nby1[0];
 
  // size dimension varibles

  int coming_fts =  faults->next_position();

  // run time variales
  double tim_allreduce = 0.0;
  double tim_v1 = 0.0;
  double tim_v2 = 0.0;
  double tim_v3 = 0.0;
  double tim_v4 = 0.0;
  double tim_mv = 0.0;
  double tim_cg = 0.0;
  double tim_ft = 0.0;


  // -------------- begining of CG ------------------------------------------//
  for(int i=0; i<nloc; i++) p_loc[i] = r_loc[i] = rhs_loc[i];
  for(int i=0; i<nloc; i++) x_loc[i] = .0;

  rddot_loc=.0; for(int i=0; i<nloc; i++) rddot_loc+=r_loc[i]*r_loc[i];
  MPI_Allreduce(&rddot_loc,&prev_rddot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 
  
  A->MultiplyVectorMPI(nloc, p_loc, array_nby1, Ap_loc);
  
  pAp_loc=.0; for(int i=0; i<nloc; i++) pAp_loc+=p_loc[i]*Ap_loc[i];  
  MPI_Allreduce(&pAp_loc, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  // ------------- offical begin of CG loops --------------------------------//
  tim_cg =-MPI_Wtime();
  int cgit=1;
  for(cgit=1;cgit<=CG_MAX_ITER;cgit++){
    const double alpha = prev_rddot / pAp;
    
    tim_v1 -= MPI_Wtime();
    for(int i=0; i<nloc; i++) x_loc[i]+= (alpha*p_loc[i]);
    tim_v1 += MPI_Wtime();

    tim_v2 -= MPI_Wtime();
    rddot_loc=0.0; 
    for(int i=0; i<nloc; i++) {
      r_loc[i]-= (alpha*Ap_loc[i]);
      rddot_loc+=r_loc[i]*r_loc[i];
    }
    tim_v2 += MPI_Wtime();
    
    tim_allreduce = -MPI_Wtime();
    MPI_Allreduce(&rddot_loc,&curr_rddot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    tim_allreduce += MPI_Wtime();
    if( curr_rddot>-ABS_TOL_SQR && curr_rddot<ABS_TOL_SQR )  break;
    double beta = curr_rddot / prev_rddot;
    
    if(cgit==coming_fts) {
      tim_ft -= MPI_Wtime();
      faults->boom(A, array_nby1, x_loc, saved_x_loc_, rhs_loc, r_loc, rddot_loc, curr_rddot, beta);
      
      row_rcvcnt = &((A->get_row_rcvcnt())[0]);
      row_displs = &((A->get_row_displs())[0]);
      disp  = row_displs[rank];

      coming_fts = faults->next_position();
      tim_ft += MPI_Wtime();
    }
    
    prev_rddot = curr_rddot;
    tim_v3 -= MPI_Wtime();
    for(int i=0; i<nloc; i++) p_loc[i]=beta*p_loc[i]+r_loc[i];
    tim_v3 += MPI_Wtime();
  
    tim_mv -= MPI_Wtime();
    A->MultiplyVectorMPI(nloc, p_loc, array_nby1, Ap_loc);
    tim_mv += MPI_Wtime();
  
    tim_v4 -= MPI_Wtime();
    pAp_loc=.0; for(int i=0; i<nloc; i++) pAp_loc+=p_loc[i]*Ap_loc[i];  
    tim_v4 += MPI_Wtime();  
    
    tim_allreduce -= MPI_Wtime();
    MPI_Allreduce(&pAp_loc, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    tim_allreduce += MPI_Wtime();
  }//end of for loop
  tim_cg += MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == ROOT_ID) printf("np %d ,ftn %d ,tot_iter %d ,tot_runtime %g ,fault_time %g ,pure_cg_time %g ,",nproc,faults->cnt(), cgit,tim_cg,tim_ft,tim_cg-tim_ft);
  double res = get_rtol_from_xloc(x_loc, faults->A4Res, faults->cnt(), saved_x_loc_.size(), &saved_x_loc_[0], faults->E4Res, faults->b4Res);
  if(rank_==ROOT_ID) printf("final_rtol %g\n",res);
  
  
  if(rank == ROOT_ID) printf("tot_runtime %g ,allreduce_time %g ,vector_time %g,mv_time %g \n",tim_cg,tim_allreduce,tim_v1+tim_v2+tim_v3+tim_v4,tim_mv);
  if(rank == ROOT_ID) printf("update_x %g ,update_r %g ,update_p %g,update_pap %g \n",tim_v1,tim_v2,tim_v3,tim_v4);
  //xloc2xglb(nloc, x_loc, row_rcvcnt, row_displs, &v_x_[0], ftn, saved_x_loc_.size(), &saved_x_loc_[0], E4Res);
  //if(rank == ROOT_ID) { printf("solution x(%d) :",(signed)v_x_.size()); for(int i=0;i<min((signed)v_x_.size(),100); i++) printf(" %g",v_x_[i]); if(v_x_.size()>100) printf("..."); printf("\n"); } 
  
  //if(!Eb) delete []Eb; Eb=nullptr;
  //if(!A4Res) delete A4Res; A4Res = nullptr;
  //if(!E4Res) delete E4Res; E4Res = nullptr;
  //if(!b4Res) delete b4Res; b4Res = nullptr;

}//end of fn solve 


// ============================================================================
// xglb <- xloc with fault recover
// ============================================================================
void MPI_ADLS_CG::xloc2xglb(int nloc, double* xloc, int *row_rcvcnt, int*row_displs, double *x, int ftn, int ftn_loc, double* saved_xloc, MtxDen*E){
  if(ftn<=0){
    MPI_Allgatherv(xloc, nloc, MPI_DOUBLE, x, row_rcvcnt, row_displs, MPI_DOUBLE, MPI_COMM_WORLD);
    return;
  }

  const int nproc = nproc_;
  const int rank  = rank_;
  // x_true and x_redundent
  double *xt_loc = new double[nloc];
  for(int i=0;       i<ftn_loc; i++)  xt_loc[i]=saved_xloc[i];
  for(int i=ftn_loc; i<nloc;    i++)  xt_loc[i]=xloc[i];


  int FTN    = (ftn/nproc)*nproc+ (ftn%nproc?nproc:0);
  int FTNloc = FTN/nproc;
  double **xxr = new double* [nproc];
  for(int i=0; i<nproc; i++)  { xxr[i] = new double[FTNloc]; for(int j=0; j<FTNloc; j++) xxr[i][j]=0; }

  for(int i=0; i<ftn_loc; i++) xxr[rank][i]=xloc[i];
  
  for(int i=0; i<nproc; i++)
    MPI_Bcast((xxr[i]), FTNloc, MPI_DOUBLE, i, MPI_COMM_WORLD);


  double *xr = new double[FTN];
  for(int i=0,j=0; i<FTNloc; i++)
    for(int pid=0; pid<nproc; pid++)
      xr[j++]=xxr[pid][i];

  for(int i=0; i<nproc; i++){
    delete []xxr[i]; xxr[i]=nullptr;
  }
  delete []xxr; xxr=nullptr;

  // xloc <- xloc + Eloc * xr
  const int K = E->cols();
  const int locBEG = row_displs[rank]*K;
  double *eloc = &(E->a_[locBEG]);
  for(int i=0; i<nloc; i++){
    const int itK = i*K;
    double t = xt_loc[i];
    for(int k=0; k<ftn; k++) 
      t+=eloc[itK+k]*xr[k];
    xt_loc[i]=t;
  }

  delete []xr; xr=nullptr;
  MPI_Allgatherv(xt_loc, nloc, MPI_DOUBLE, x, row_rcvcnt, row_displs, MPI_DOUBLE, MPI_COMM_WORLD);
  delete []xt_loc; xt_loc=nullptr;
}

// ============================================================================
// residual
// ============================================================================
double MPI_ADLS_CG::get_rtol_from_xloc(double *xloc, MtxSpMPI *A, int ftn, int ftn_loc, double* saved_xloc,  MtxDen*E, MtxDen*b){
  int nloc = A->rows_loc();
  int* row_rcvcnt = &( (A->get_row_rcvcnt())[0] );
  int* row_displs = &( (A->get_row_displs())[0] );
  double *x = &v_x_[0];
  xloc2xglb(nloc, xloc, row_rcvcnt, row_displs, x, ftn, ftn_loc, saved_xloc, E);

  double   *bloc = &(b->a_[row_displs[rank_]]); 
  double *Axloc = new double[nloc];
  A->MultiplyVector(nloc, x, Axloc);

  for(int i=0; i<nloc; i++) Axloc[i]-=bloc[i];

  double nomi_loc=0,deno_loc=0;
  for(int i=0; i<nloc; i++) nomi_loc+= Axloc[i]*Axloc[i];
  for(int i=0; i<nloc; i++) deno_loc+=  bloc[i]* bloc[i];

  double nomi, deno;
  MPI_Allreduce(&nomi_loc,&nomi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&deno_loc,&deno, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return sqrt(nomi/deno);
}



// ============================================================================
// xglb <- xloc with fault recover
// ============================================================================
/*
void MPI_ADLS_CG::gather_solution(bool ft_happens, int ftn, MtxDen* E){

  int nloc = A_->rows_loc(); 
  double *x_loc = new double[nloc];
  for(int i=0; i<nloc; i++) x_loc[i] = v_x_loc_[i];

  if(ft_happens){
    int ftn_loc = ftn/nproc_;
    int disp = n_disps_[rank_];

    double *xr_loc = new double[ftn_loc];
    for(int i=0; i<ftn_loc; i++) { xr_loc[i]=x_loc[i]; x_loc[i]=bkuped_ftx_loc[i];}
    double *xr = new double[ftn];
    MPI_Allgather(xr_loc, ftn_loc, MPI_DOUBLE, xr, ftn_loc, MPI_DOUBLE, MPI_COMM_WORLD);
    delete []xr_loc; xr_loc=nullptr;
    double *e_loc = &(E->a_[disp*k_]);
    for(int i=0; i<nloc; i++){
      for(int k=0; k<k_; k++) {
        x_loc[i]+= e_loc[i*k_+k]*xr[k];
      }
    }
    delete []xr; xr=nullptr;
  }
  double *xglb = &v_x_[0];
  MPI_Allgatherv(x_loc, nloc_, MPI_DOUBLE, xglb, &(n_lens_[0]), &(n_disps_[0]), MPI_DOUBLE, MPI_COMM_WORLD);
  delete []x_loc; x_loc=nullptr;

}
*/


 /*
void MPI_ADLS_CG::get_res(MtxSpMPI*org, double* bloc, double* Axloc){

  double *x = &v_x_[0];
  org->Multiply_loc(x, Axloc);
  for(int i=0; i<nloc_; i++) Axloc[i]-=bloc[i];

  double nomi_loc=0,deno_loc=0;
  for(int i=0; i<nloc_; i++) nomi_loc+= Axloc[i]*Axloc[i];
  for(int i=0; i<nloc_; i++) deno_loc+=  bloc[i]* bloc[i];

  double nomi, deno;
  MPI_Reduce(&nomi_loc,&nomi, 1, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
  MPI_Reduce(&deno_loc,&deno, 1, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);

  if(rank_==ROOT_ID) printf("%g\n",sqrt(nomi/deno));

}

*/


// ============================================================================
// Construction 
// ============================================================================
MPI_ADLS_CG::MPI_ADLS_CG(const string &f_A, const string &f_E, const string &f_rhs, const string& ftfa_base, vector<int>&vfts,  int rank, int nproc)
: rank_(rank),
  nproc_(nproc),
  file_A_(f_A),
  file_E_(f_E),
  file_rhs_(f_rhs),
  A_(nullptr),
  b_(f_rhs)
{

  if(f_A.size()<4 || f_A.substr(f_A.length()-5,4)!=".mtx"){
    A_    = new MtxSpMPI(f_A, "mm", rank, nproc);
  }
  else
    A_    = new MtxSpMPI(f_A, "binary", rank, nproc);

  int nloc = A_->rows_loc();
  int n = A_->rows();
  
  v_p_loc_.assign(nloc,.0);
  v_Ap_loc_.assign(nloc,.0);
  v_x_loc_.assign(nloc,.0);
  v_r_loc_.assign(nloc,.0);

  //v_p_.assign(n, .0);
  v_x_.assign(n, .0);

  faults = new LSFaultMPI(f_A, f_E, f_rhs, vfts, ftfa_base, rank, nproc);
  return;
}


// ============================================================================
// DeConstruction
// ============================================================================
MPI_ADLS_CG::~MPI_ADLS_CG(){ 
  if(A_) delete A_; A_=nullptr; 
  //if(Atut_) delete Atut_; Atut_=nullptr; 
  if(faults) delete faults; faults=nullptr;
}



// ============================================================================
// dump
// ============================================================================
void MPI_ADLS_CG::dump(){
//  fprintf(stdout,"dump() in Class MPI_ADLS_CG\n");
//  fprintf(stdout,"rank %d, nproc %d, n %d, k %d, nloc %d\n",rank_, nproc_, n_, k_, nloc_);
  
//  fprintf(stdout,"n_lens (%d) : ",(int)n_lens_.size());
//  for(auto x: n_lens_) fprintf(stdout, "%d ",x);
//  fprintf(stdout, "\n");

//  fprintf(stdout,"n_disps (%d) : ",(int)n_disps_.size());
//  for(auto x: n_disps_) fprintf(stdout, "%d ",x);
//  fprintf(stdout, "\n");

  //if(rank_==gdbid) {
  //  A_.dump();
  //  b_.dump();
  //}
  
//  fprintf(stdout,"end of MPI_ADLS_CG dump()\n");

}

// ============================================================================
// Error
// ============================================================================
void MPI_ADLS_CG::error(const string&s1){ 
  fprintf(stdout,"Error in class MPI_ADLS_CG, with message \"%s\"\n",s1.c_str());
  exit(1);
}

// ============================================================================
// Error
// ============================================================================
void MPI_ADLS_CG::error(const string&s1,const string&s2){ 
  fprintf(stdout,"Error in class MPI_ADLS_CG, with message \"%s %s\"\n",s1.c_str(),s2.c_str());
  exit(1);
}


//#ifdef DEBUG_ADPTLS_CPP
//  MPI_Barrier(MPI_COMM_WORLD);
//  if(rank_!=0){int msg=0; MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);}
//    printf("%d/%d %s",rank, nproc, ss.str().c_str());
//  ss.str("");
//  if(rank_+1!=nproc_) { int msg=0; MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD); }
//  MPI_Barrier(MPI_COMM_WORLD);
//#endif



