#ifndef MTXDEN_HPP
#define MTXDEN_HPP
/************************************************************************
  > File Name: MtxDen.hpp
  > Author: xin cheng
  > Descriptions: class of dense matrix.
  > Created Time: before 2020
 ************************************************************************/

#include <vector>
#include <string>

using std::vector;
using std::string;
// ============================================================================
// Dense Matrix
// ============================================================================
class MtxDen
{
public:
    MtxDen(const string &filename) : m_(0), n_(0), a_({}) { do_read_matrixmarket(filename); };

public:
    virtual ~MtxDen(){};

public:
    MtxDen(MtxDen &&) = delete;
    MtxDen(const MtxDen &) = delete;
    MtxDen &operator=(MtxDen &&) = delete;
    MtxDen &operator=(const MtxDen &) = delete;

public:
    void MultiplyMatrix(int K, double *X, double *Y);

public:
    void TransMultiplyMatrix(int K, double *X, double *Y); //k is number of col of Y

public:
    int rows() const { return m_; }
    int cols() const { return n_; }

private:
    int m_, n_;

public:
    vector<double> a_;

public:
    virtual void dump(int indent = 0);
    virtual string toString(){ return ""; }

protected:
    virtual void do_read_matrixmarket(string const & filename);
};
#endif
