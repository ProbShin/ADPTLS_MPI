#ifndef MTXSPMPIA2A_HPP
#define MTXSPMPIA2A_HPP
/************************************************************************
  > File Name: inc/mtxSpMpiA2a.hpp
  > Author: xin cheng
  > Descriptions: class of distributed spares matrix (MPI) supported 
  >               MatVec multiplication using MPI_all2all method. 
  > Created Time: 2021 Aug 1
 ************************************************************************/
#include "MtxSpMpi.hpp"
#include <vector>

using std::vector;
class MtxSpMpiA2a : public MtxSpMpi
{
public:
    MtxSpMpiA2a(int rnk, int np) : MtxSpMpi(rnk, np) {}
    MtxSpMpiA2a(const string &file_name, int rnk, int np) : MtxSpMpiA2a(rnk, np) { do_read_matrixmarket(file_name); }
    virtual ~MtxSpMpiA2a() {}

public:
    virtual void MultiplyVector_all2allv(double *xloc, double *yloc);
    virtual string toString();

private:
    virtual void do_read_matrixmarket(const string &file_name);

protected:
    vector<double> send_buf_;
    vector<int> send_buf_map_i2nzc_;
    vector<int> send_counts_;
    vector<int> send_displs_;

    vector<double> recv_buf_;
    vector<int> recv_buf_map_j2index_;
    vector<int> recv_counts_;
    vector<int> recv_displs_;
};

#endif
