#ifndef MTXSPMPIA2A_HPP
#define MTXSPMPIA2A_HPP
/************************************************************************
  > File Name: inc/mtxSpMpiA2a.hpp
  > Author: xin cheng
  > Descriptions: class of distributed spares matrix (MPI) supported 
  >               MatVec multiplication using MPI_all2all method. 
  > Created Time: 2021 Aug 1
 ************************************************************************/
#include "mtx_basic.hpp"
#include <vector>

class MtxSpMpiA2a : public MtxSpMPI {
public:
    MtxSpMpiA2a(int rnk, int np) : MtxSpMPI(rnk, np) {}
    MtxSpMPIA2a(const string&file_name, int rnk, int np) : MtxSpMpiA2a(rnk, np) {
        doReadMtxSpecial4MpiA2a(file_name);
    } 
    virtual ~MtxSpMpiA2a(){}

public:
    virtual void MultiplyVector_all2allv(int nrows,  double* xloc, double *yloc);

private:
    void doReadMtxSpecial4MpiA2a(const string& filename);

protected:
    vector<int>         send_buf_;
    vector<int>         send_buf_map_i2nzc_; 
    vector<int>         send_counts_;
    vector<int>         send_displs_;

    vector<int>         recv_buf_;
    vector<int>         recv_buf_map_j2index_;
    vector<int>         recv_counts_;
    vector<int>         recv_displs_;
    
};

#endif
