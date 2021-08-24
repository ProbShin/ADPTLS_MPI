
#include "mpi.h"
#include "def.hpp"
#include "adptls.hpp"
#include "MtxSpMpiA2a.hpp"


#include <cstring>
#include <iostream>
#include <cstdio>  /* stderr */
#include <cstdlib> /* exit */
#include <fstream> /* fstream */
#include <sstream>

#include <cmath> /* fabs */

using namespace std;

//int gdbid=0;
//bool gdbflag=false;

template <class T> void dump(vector<T> &v, string const &vname, int rank, int nproc);


int main(int argc, char const *argv[]){
  int nproc, rank;
  MPI_Init(&argc, (char***) &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  string file_A, file_E, file_rhs, ftfa;
  vector<int> vfts;
 
  if(argc<=1) { printf("\nusage: check run.sh\n"); MPI_Finalize(); exit(1); }
  for(int i=1; i<argc; i++){
    if(     !strcmp(argv[i], "-fA"  )) file_A = argv[++i];
    else if(!strcmp(argv[i], "-fE"   )) file_E = argv[++i];
    else if(!strcmp(argv[i], "-rhs"  )) file_rhs = argv[++i];
    else if(!strcmp(argv[i], "-fts"))  for(int j=i+1; j<argc && argv[j][0]!='-'; j++, i++) vfts.push_back(atoi(argv[j]));
    else if(!strcmp(argv[i], "-ftbase"))  ftfa = argv[++i];
    //else if(!strcmp(argv[i], "-aug"   )) file_aug = argv[++i];
    //else if(!strcmp(argv[i], "-fAtut"  )) file_aug_faulted = argv[++i];
    //else if(!strcmp(argv[i], "-theox")) file_theo= argv[++i];

    //else if(!strcmp(argv[i], "-gdb"))  gdbid=atoi(argv[++i]);
    else { fprintf(stderr,"Warning! unused input argument\"%s\"",argv[i]); }
  }

  {
    auto tim = -MPI_Wtime();
    
    MtxSpMpiA2a A(file_A, rank, nproc);
    
    tim += MPI_Wtime();
    if(rank==ROOT_ID) {
      //cout<<A.toString()<<"\n";
      cout<<"Matrix construction time "<< tim<<"\n";
    }

    int num_col_loc = 0;
    {
      int col_glb = A.cols();      
      vector<int> cols_rcvcnt(nproc, col_glb/nproc);
      for(int i=0; i < col_glb%nproc; i++) cols_rcvcnt[i]++;
      num_col_loc = cols_rcvcnt[rank];
      vector<int> cols_displs(nproc+1, 0);
      for(int i=1; i <= nproc; i++) cols_displs[i] = cols_displs[i-1] + cols_rcvcnt[i-1];
    }
    vector<double> xloc(num_col_loc, 1.0);
    

    vector<double> yloc(num_col_loc, 0);

    A.MultiplyVector_all2allv(&xloc[0], &yloc[0]);
    
    vector<double> y;
    if(rank == 0) y.assign(A.rows(), .0);
    MPI_Gatherv(&yloc[0], A.get_row_rcvcnt()[rank], MPI_DOUBLE,
                   &y[0], &(A.get_row_rcvcnt()[0]), &(A.get_row_displs()[0]), MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);
    if(rank == 0) {
      std::stringstream ss;
      ss<<"\n\n\ny("<<y.size()<<"):[";
      for(auto e : y) ss << e << ",";
      ss<<"]\n\n\n\n" ;
      std::cout<<ss.str()<<std::flush;
    }



  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Finalize();
  return  0;
}

template <class T>
void dump(vector<T> &v, string const &vname, int rank, int nproc)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int msg = 0;
  if (rank != 0)
    MPI_Recv(&msg, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int const MaxNumElmToShow = 100;

  stringstream ss;
  ss << "\nrank: " << rank << " "
     << (vname.empty() ? "vec" : vname) << "(" << v.size() << "):[";
  for (auto i = 0; i < v.size() && i < MaxNumElmToShow; i++)
    ss << v[i] << ",";
  ss << (v.size() > MaxNumElmToShow ? "...]" : "]") << "\n";
  cerr << ss.str() << std::flush;

  if (rank + 1 != nproc)
    MPI_Send(&msg, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

/*


  double rtime;
  double iotime;
  double j_start, j_end, j_total = 0.0;

  if( argc!=5){
    if(rank==ROOT_ID){
      fprintf(stderr,  ACRED"usage: %s "ACGREEN"<inAtute.mtx> <inbtute.mtx>"
          " <inE> "ACRESET"\n", argv[0]);
    }
    MPI_Finalize();
    exit(1);
  }

  string fAtut(argv[1]), fbtut(argv[2]), fE(argv[3]);
  double *E(NULL), 
         *btut(NULL),  
         *x(NULL); 
  int    ncpnr(0), nc(0), nr(0);

  double tim_t1(.0), tim_t2(.0), tim_t3(.0), 
         tim_t4(.0), tim_t5(.0);

  tim_t1 = -omp_get_wtime();
  CSRMatrix Atut(fAtut, rank, ROOT_ID);
  tim_t1 += omp_get_wtime();
  
  tim_t2 = -omp_get_wtime();
  col_domain_mem(fbtut, ncpnr, btut, rank, ROOT_ID);
  tim_t2 += omp_get_wtime();
  
  tim_t3  = -omp_get_wtime();
  row_domain_mem(fE, nc, nr, E, rank, ROOT_ID);
  tim_t3  += omp_get_wtime();

  assert(nc+nr == ncpnr);

  if(rank==ROOT_ID)
    x = new double[nc]();

  if(rank==ROOT_ID){
    printf("MPI_Adaptive_LinearSolver\n"
        "MPI_Comm_size %d\n"
        "arg1: Atut: %s  IOTime %f\n"
        "arg2: btut: %s  IOTime %f\n"
        "arg3: E   : %s  IOTime %f\n"
        "IOTime(total): %f\n"
        "nc,nr,nc+nr: %d, %d, %d\n",
        nproc,
        fAtut.c_str(), tim_t1,
        fbtut.c_str(), tim_t2,
        fE.c_str(),    tim_t3,
        tim_t1+tim_t2+tim_t3,
        nc,nr,ncpnr
        );
  }

  MPI_Barrier(MPI_COMM_WORLD);
  adaptive_cg_solver(rank, nproc,  
      Atut, btut, E, x );

  if(x    ) delete []x;
  if(btut ) delete []btut;
  if(E    ) delete []E;
  
  MPI_Finalize();
}

*/

/*

   MPI_Gather(x_loc, np, MPI_DOUBLE, x, np, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);  

   if(rank==ROOT_ID)
   printf("wocao final rtol is%f\n",rtol_calculate(m,nc,nr,nproc,xrr,E,Avalarr,Acolind,Arbegin,b));

   myCG(m, valarr, colind, rbegin, b_loc, x_loc, rank, nproc,\
   nc, nr, xrr, E, Avalarr, Acolind,Arbegin,b);

   if(rank==ROOT_ID)
   fprintf(stdout,"total run time "ACRED"%f"ACRESET"\n",-(rtime-=omp_get_wtime()));


   read_dense_mm_to_dense_col_domain_newmem(argv[2], tm, tn, brr);
   xrr = new double[m];
   if(tm!=m || tn!=1){
   fprintf(stderr, "b:%s should be %dx1 mtx but acutal get %dx%d\n",argv[2], m, tm,tn);
   exit(2);
   }



//for rtol
int    nc=0,nr=0;
double *Eorg=NULL;
double *Aorg_valarr=NULL;
int    *Aorg_colind=NULL, *Aorg_rbegin=NULL;
int    Aorg_nnz=0;
double *borg=NULL;

double *borg_loc = NULL;
if(rank==ROOT_ID){
string fE(argv[3]), fA(argv[4]), fb(argv[5]);
//printf("%s\n",fE.c_str());
//printf("%s\n",fA.c_str());
//printf("%s\n",fb.c_str());
rtol_prepare(fE,fA,fb,nc,nr,Eorg,Aorg_valarr,Aorg_colind,Aorg_rbegin, Aorg_nnz,borg);

}


MPI_Bcast  (Aorg_valarr, nnz, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
MPI_Bcast  (Aorg_colind, nnz, MPI_INT,    ROOT_ID, MPI_COMM_WORLD);
MPI_Bcast  (Aorg_rbegin, m+1 , MPI_INT,    ROOT_ID, MPI_COMM_WORLD);

MPI_Scatter(borg, mp, MPI_DOUBLE, borg_loc, mp, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);





int    *colind=NULL; //new int [annz];   for csr
int    *rbegin=NULL; //new int [m+1];    for csr
double *valarr=NULL; //new double[annz]; for csr

double *brr   =NULL; //new double[m];    b
double *b_loc =NULL; //new double[m/p]   local b

double *xrr   =NULL; // *&xrr = brr ;     x
double *x_loc =NULL; // *&xrr = brr ;     x
int m, nnz;          //mtx size; actual nnz; number of right hand sides
int mp;              //m/p -> mm



MPI_Barrier(MPI_COMM_WORLD);

if(rank==ROOT_ID)
iotime = omp_get_wtime();

if(rank==ROOT_ID)
{
  int tn;
  read_sparse_symm_mm_to_csr_newmem( argv[1], m, tn,
      nnz, valarr, colind, rbegin );    
  if(m!=tn || nnz<=0){
    fprintf(stderr, "A:%s should be symmetric but acutal get %dx%d nnz:%d", argv[1], m, tn, nnz);
  }
}

if(rank==ROOT_ID)
{
  int tm,tn;
  read_dense_mm_to_dense_col_domain_newmem(argv[2], tm, tn, brr);
  xrr = new double[m];
  if(tm!=m || tn!=1){
    fprintf(stderr, "b:%s should be %dx1 mtx but acutal get %dx%d\n",argv[2], m, tm,tn);
    exit(2);
  }
}

MPI_Bcast(&m,    1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
MPI_Bcast(&m,    1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
MPI_Bcast(&nnz,  1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);

mp = m/nproc;

if(valarr==NULL ) valarr= new double[nnz];
if(colind==NULL ) colind= new int   [nnz];
if(rbegin==NULL ) rbegin= new int   [m+1 ];

if(b_loc ==NULL ) b_loc = new double[mp ];
if(x_loc ==NULL ) x_loc = new double[mp ];

fill(x_loc, x_loc+mp,0.0);
fill(b_loc, b_loc+mp,0.0);

MPI_Bcast  (valarr, nnz, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
MPI_Bcast  (colind, nnz, MPI_INT,    ROOT_ID, MPI_COMM_WORLD);
MPI_Bcast  (rbegin, m+1 , MPI_INT,    ROOT_ID, MPI_COMM_WORLD);

MPI_Scatter(brr, mp, MPI_DOUBLE, b_loc, mp, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

if(rank==ROOT_ID)
  fprintf(stdout, "file_io(preparation) takes "ACRED"%lg(s)"ACRESET"\n", -(iotime-=omp_get_wtime()));


  MPI_Barrier(MPI_COMM_WORLD);
if(rank==ROOT_ID)
  rtime = omp_get_wtime();


  myCG(m, valarr, colind, rbegin, b_loc, x_loc, rank, nproc,\
      nc, nr, xrr, E, Avalarr, Acolind,Arbegin,b);

if(rank==ROOT_ID)
  fprintf(stdout,"total run time "ACRED"%f"ACRESET"\n",-(rtime-=omp_get_wtime()));



  MPI_Gather(x_loc, mp, MPI_DOUBLE, xrr, mp, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);  
if(rank==ROOT_ID)
  printf("wocao final rtol is%f\n",rtol_calculate(m,nc,nr,nproc,xrr,E,Avalarr,Acolind,Arbegin,b));


  delete []valarr;
  delete []colind;
  delete []rbegin;
  delete []b_loc;
  delete []x_loc;



  }




*/
/*

void rtol_prepare(string& fE, string& fA, string& fb, \
    int &nc, int &nr, double* &E, double* &Avalarr, \
    int* &Acolind, int* &Arbegin, int& Annz, double*&b);

void rtol_prepare(string& fE, string& fA, string& fb, \
    int &nc, int &nr, double* &E, double*& Avalarr, \
    int* &Acolind, int* &Arbegin, int& Annz, double *&b){ 
  //int    nc,nr;
  //double *E;
  //double *Avalarr;
  //int    *Acolind, *Arbegin;
  //int    Annz;
  //double *b;
  {
    int t1,t2,t3,t4,t5,t6,tnnz;
    //read_sparse_non_symm_mm_to_dense_row_domain_newmem(fE.c_str(), t1, t2, tnnz, E);
    read_sparse_symm_mm_to_csr_newmem        (fA.c_str(), t3,t4, Annz, \
        Avalarr, Acolind, Arbegin);
    read_dense_mm_to_dense_col_domain_newmem (fb.c_str(), t5,t6, b);

    nc=t1;
    nr=t2;
    if( t1!=nc ||
        t2!=nr ||
        t3!=nc ||
        t4!=nc ||
        t5!=nc ||
        t6!=1  ) {
      fprintf(stderr,"error in E,A,b\n");
      //fprintf(stderr, "matrix size is not match E:%s A:%s b:%s should be (%dx%d, %dx%d, %dx1) however we get (%dx%d %dx%d %dx%d)\n", fE,fA,fb, nc,nr,nc,nc,nc, t1,t2,t3,t4,t5,t6);
      exit(2);
    }
  }
}//end of function rtol_prepare()

*/






