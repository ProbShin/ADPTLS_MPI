/************************************************************************
  > File Name: MtxDen.cpp
  > Author: xin cheng
  > Descriptions: Implementation of class MtxDen
  > Created Time: Sun Dec 31 15:52:40 2017
 ************************************************************************/
#include "MtxDen.hpp"

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using std::getline;
using std::ifstream;
using std::stringstream;
using std::to_string;
using std::runtime_error;

// ============================================================================
// constructor of a dense matrix
// ----------------------------------------------------------------------------
// e.g. augmented sys E is a n by k dense matrix
// ============================================================================
void MtxDen::do_read_matrixmarket(string const &file_name)
{
    // bool bDens = false;

    if (file_name.empty())
        throw runtime_error("file name is empty.\n");
    ifstream fp(file_name);
    if (!fp.is_open())
        throw runtime_error(string("cannot open file '") + file_name + "'\n");

    string line, word;
    stringstream ss;
    {
        // check the format
        getline(fp, line);
        ss.str(line);
        string word1, word2, word3;
        if (!(ss >> word) || (word != "%%MatrixMarket") || !(ss >> word) ||
            (word != "matrix") || (ss >> word1 >> word2 >> word3))
            throw runtime_error(string("MatrixMarket file '") + file_name +
                                "' format error with headline '" + line + "'\n");

        if (word1 != "array")
            throw runtime_error(string("MatrixMarket file '") + file_name +
                                " is not a dense matrix.\n");

        if (word2 != "real" && word2 != "integer")
            throw runtime_error(string("MatrixMarket file '") + file_name +
                                " with the qualifier '" + word2 +
                                "' is not supported.\n");

        if (word3 != "general")
            throw runtime_error(
                string("MatrixMarket file '") + file_name + " with the qualifier '" +
                word3 + "' is not supported. Only support 'general' for now.\n");
    }

    while (getline(fp, line))
    {
        if (line.empty() || line[0] == '%')
            continue;
        break;
    }

    int r, c, expected_num_lines = 0;
    ss.clear();
    ss.str(line);
    if (!(ss >> r >> c))
        throw runtime_error(
            string("could not read the matrx dimension from the line '") + line +
            "' of matrix file '" + file_name + "'");

    expected_num_lines = r * c;

    m_ = r;
    n_ = c;
    a_.assign(m_ * n_, .0);

    // read the matrix
    int num_entries_readed = 0;
    while (getline(fp, line) && num_entries_readed <= expected_num_lines)
    {
        if (line.empty() || line[0] == '%')
            continue;
        ss.str(line);
        ss >> a_[num_entries_readed++];
    }

    if (num_entries_readed != expected_num_lines)
        throw runtime_error(
            string("expected to read ") + to_string(expected_num_lines) +
            "' entries but only read " + to_string(num_entries_readed) +
            "lines. The file '" + file_name + "' could be truncated or damaged.\n");

    fp.close();
}

// ============================================================================
// Y<- A*X       Y[ik]=A[ij]*X[jk]
// ----------------------------------------------------------------------------
// X is a row_domain, n_ by l matrix
// y is a row_domain, m_ by l matrix
// since row domain, uses for(ikj)
// ============================================================================
void MtxDen::MultiplyMatrix(int L, double *X, double *Y)
{
    for (int i = 0; i < m_; i++)
    {
        const int itL = i * L;
        const int itN = i * n_;
        for (int k = 0; k < L; k++)
            Y[itL + k] = 0;
        for (int j = 0; j < n_; j++)
        {
            const int jtL = j * L;
            const double aij = a_[itN + j];
            for (int k = 0; k < L; k++)
                Y[itL + k] += aij * X[jtL + k];
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
void MtxDen::TransMultiplyMatrix(int L, double *X, double *Y)
{
    // fprintf(stderr,"MtxDen m_%d,n_%d\n",m_,n_);

    for (int i = 0; i < n_; i++)
    { // for i column of A,
        const int itL = i * L;
        for (int k = 0; k < L; k++)
        {
            Y[itL + k] = 0;
            // fprintf(stderr,"Y[i%d*L%d=itL%d+k%d]=0 ",i,L,itL,k);
        }
        // fprintf(stderr,"\n");
        for (int j = 0; j < m_; j++)
        {
            const int jtL = j * L;
            const int jtN = j * n_;
            const double aji = a_[jtN + i];
            // fprintf(stderr,"aji=a[j%d*N*%d=jtN%d+i]=%lg\n",j,n_,jtM, aji);
            for (int k = 0; k < L; k++)
            {
                Y[itL + k] += aji * X[jtL + k];
                // fprintf(stderr,"Y[%d]+=aji*X[%d] ",itL+k, jtL+k);
            }
            // fprintf(stderr,"\n");
        }
    }
}

// ============================================================================
// dump
// ============================================================================
void MtxDen::dump(int indent)
{
    fprintf(stderr, "\n\nMtxDen::dump()\n");
    for (int i = 0; i < indent; i++)
        fprintf(stdout, " ");
    fprintf(stderr, "rows: %d, cols: %d\n", m_, n_);

    for (int i = 0; i < indent; i++)
        fprintf(stdout, " ");
    fprintf(stderr, "a_(%d):", (signed)a_.size());
    for (auto x : a_)
        fprintf(stdout, " %lg", x);
    fprintf(stderr, "\n\n");
}
