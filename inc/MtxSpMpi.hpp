#ifndef MTXSPMPI_HPP
#define MTXSPMPI_HPP
/************************************************************************
  > File Name: MtxSp.hpp
  > Author: xin cheng
  > Descriptions: class of spares matrix supported MPI (Message Passing
  >               Interface) So that large matrix divided into seperated machines.
  > Created Time: before 2020
 ************************************************************************/

#include "MtxSp.hpp"
#include <mpi.h>
#include <vector>

using std::string;
using std::vector;

// ============================================================================
// MPI Distributed Matrix
// ============================================================================
class MtxSpMpi : public MtxSp
{
public:
    MtxSpMpi(int rnk, int np) : MtxSp(), rank_(rnk), nproc_(np), row_rcvcnt_({}), row_displs_({}),
                                num_rows_loc_(0), num_cols_loc_(0), num_rows_glb_(0), num_cols_glb_(0) {}

public:
    MtxSpMpi(const string &file_name, int rnk, int np) : MtxSpMpi(rnk, np) { do_read_matrixmarket(file_name); }
    virtual ~MtxSpMpi(){};
    MtxSpMpi(MtxSpMpi &&) = delete;
    MtxSpMpi(const MtxSpMpi &) = delete;
    MtxSpMpi &operator=(MtxSpMpi &&) = delete;
    MtxSpMpi &operator=(const MtxSpMpi &) = delete;

public:
    void MultiplyVector_Allgatherv(int nrows, double *xglb, double *yloc, double *yglb, int *recvcounts, int *displs);
    void MultiplyMatrix_Allgatherv(int nrows, int K, double *xglb, double *yloc, double *yglb, int *recvcounts, int *displs);

public:
    virtual int rows() const { return num_rows_glb_; }
    virtual int cols() const { return num_cols_glb_; }
    virtual int rows_loc() const { return num_rows_loc_; }
    virtual int cols_loc() const { return num_cols_loc_; }

    vector<int> &get_row_displs() { return row_displs_; }
    vector<int> &get_row_rcvcnt() { return row_rcvcnt_; }

public:
    virtual void do_read_matrixmarket(string const &file_name);
    virtual void dump();
    virtual void dump(vector<int> &v) const { dump(v, ""); };
    virtual void dump(vector<int> &v, string const &vname) const;
    virtual string toString();

protected:
    const int rank_;
    const int nproc_;

    vector<int> row_rcvcnt_;
    vector<int> row_displs_;

protected:
    int num_rows_loc_, num_cols_loc_;
    int num_rows_glb_, num_cols_glb_;
};

#endif
