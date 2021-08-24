#ifndef MTXSP_HPP
#define MTXSP_HPP
/************************************************************************
  > File Name: MtxSp.hpp
  > Author: xin cheng
  > Descriptions: class of spares matrix, the matrix is contains both upper and lower part
  >               with Compressed Sparse Row (CSR) format.
  > Created Time: before 2020
 ************************************************************************/

#include <vector>
#include <string>
using std::string;
using std::vector;

// ============================================================================
// Sparse Symmetric Matrix, stored both upper and lower part
// ============================================================================
class MtxSp
{
public:
    MtxSp() : num_rows_(0), num_cols_(0), ia_({}), ja_({}), a_({}){};

public:
    MtxSp(const string &file_name) : MtxSp() { do_read_matrixmarket(file_name); }
    virtual ~MtxSp(){};
    MtxSp(MtxSp &&) = delete;
    MtxSp(const MtxSp &) = delete;
    MtxSp &operator=(MtxSp &&) = delete;
    MtxSp &operator=(const MtxSp &) = delete;

public:
    virtual int rows() const { return num_rows_; }
    virtual int cols() const { return num_cols_; }
    virtual void rows(int r) { num_rows_ = r; }
    virtual void cols(int c) { num_cols_ = c; }
    vector<int> &get_ia() { return ia_; }
    vector<int> &get_ja() { return ja_; }
    vector<double> &get_a() { return a_; }

    double wocao_a(int i) { return a_[i]; }

    void MultiplyVector(int nrows, double *x, double *y);
    void MultiplyMatrix(int nrows, int ncols, double *x, double *y);

public:
    virtual void do_read_matrixmarket(string const &filename);
    virtual void dump();
    virtual string toString();

protected:
    vector<int> ia_;
    vector<int> ja_;
    vector<double> a_;

private:
    int num_rows_;
    int num_cols_;
};

#endif
