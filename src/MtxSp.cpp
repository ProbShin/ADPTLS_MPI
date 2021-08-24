/************************************************************************
  > File Name: MtxSp.cpp
  > Author: xin cheng
  > Descriptions: implementation of MtxSp
  > Created Time: Sun Dec 31 2017
 ************************************************************************/
#include "MtxSp.hpp"
#include <unordered_map>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using std::cerr;
using std::cout;
using std::ifstream;
using std::pair;
using std::runtime_error;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::vector;
using std::to_string;
using std::sort;

// read a sparse matri from matrix market file
void MtxSp::do_read_matrixmarket(string const &file_name)
{

    num_rows_ = 0;
    num_cols_ = 0;
    a_.clear();
    a_.shrink_to_fit();
    ia_.clear();
    ia_.shrink_to_fit();
    ja_.clear();
    ja_.shrink_to_fit();

    bool bDens = false;
    bool bSymm = false;

    if (file_name.empty())
        throw runtime_error("file name is empty.\n");

    ifstream fp(file_name);
    if (!fp.is_open())
        throw runtime_error(string("cannot open file'") + file_name + "'.\n");

    string line, word;
    stringstream ss;
    getline(fp, line);
    ss.str(line);
    { // check the format
        string word1, word2, word3;
        if (!(ss >> word) || (word != "%%MatrixMarket") || !(ss >> word) || (word != "matrix") || (ss >> word1 >> word2 >> word3))
            throw runtime_error(
                string("MatrixMarket file '") + file_name + "' format error with headline '" + line + "'\n");

        if (word1 == "array")
        {
            bDens = true;
            cerr << "Warning! MatrixMarket file '" << file_name << "' contains a dense matrix while expect a sparse one.\n";
        }
        else if (word1 != "coordinate")
            throw runtime_error(string("MatrixMarket file '") + file_name + "' is neigher 'array' nor 'coordinate', with head line '" + line + "'\n");

        if (word2 != "real")
        {
            if (word2 == "integer")
                cerr << "Warning! MatrixMarket file '" << file_name << " is a integer matrix, the integer will be stored as a double float.\n";
            else
                throw runtime_error(string("The matrix of MatrixMarket file '") + file_name + "' is of type '" + word2 + "' which is not supported. The only supported format is 'real' or 'integer'.\n");
        }

        if (word3 != "general")
        {
            bSymm = true;
            if (word3 == "skew-symmetirc" || word3 == "Hermitian")
                cerr << "Warning! The '" << word3 << "' matrix of MatrixMarket file '" << file_name << "' of qualifier '" << word3 << "' will be read as the 'symmetric' graph.\n";
            else if (word3 != "symmetric")
                throw runtime_error(string("The '") + word3 + "' matrix of file '" + file_name + "' is not supported.\n");

            if (bDens)
                throw runtime_error(string("Does not support read BOTH 'dense' and 'symmetric' matrix of matrixmarket format, however file '") + file_name + "' is.\n");
        }
    }

    while (getline(fp, line))
    {
        if (line.empty() || line[0] == '%')
            continue;
        break;
    }

    // read dimension
    int r, c, expected_num_lines = 0;
    ss.clear();
    ss.str(line);
    if (!(ss >> r >> c) || (!bDens && !(ss >> expected_num_lines)))
        throw runtime_error(string("could not read the matrx "
                                   "dimension from the line '") +
                            line + "' of matrix file '" + file_name + "'");
    if (bDens)
        expected_num_lines = r * c;
    if (bSymm && r != c)
        throw runtime_error(string("symmetric matrix '") + file_name + "' number_of_rows (" + to_string(r) + ") != number_of_cols (" + to_string(c) + ").\n");

    num_rows_ = r;
    num_cols_ = c;

    // read the matrix
    int num_entries_readed = 0;
    unordered_map<int, vector<pair<int, double>>> G;
    int i, j;
    double v;
    ss.clear();
    i = 0, j = 0, v = .0;
    if (bDens)
    { // matrix is dense
        while (getline(fp, line) && num_entries_readed <= expected_num_lines)
        {
            if (line.empty() || line[0] == '%')
                continue;
            num_entries_readed += 1;
            ss.str(line);
            ss >> v;
            G[i].emplace_back(j, v);
            if (++i >= num_rows_)
            {
                i = 0;
                j += 1;
            }
        }
    }
    else if (!bSymm)
    { // matrix is sparse non-symmetric
        while (getline(fp, line) && num_entries_readed <= expected_num_lines)
        {
            if (line.empty() || line[0] == '%')
                continue;
            num_entries_readed += 1;
            ss.str(line);
            ss >> i >> j >> v;
            i -= 1;
            j -= 1;
            G[i].emplace_back(j, v);
        }
    }
    else
    { // matrix is sparse symmetric
        while (getline(fp, line) && num_entries_readed <= expected_num_lines)
        {
            if (line.empty() || line[0] == '%')
                continue;
            num_entries_readed += 1;
            ss.str(line);
            ss >> i >> j >> v;
            if (i < j)
                throw runtime_error(string("read a upper diagonal entry '") + to_string(i) + ", " + to_string(j) + ", " + to_string(v) + "' around " + to_string(num_entries_readed) + "' lines of file '" + file_name + "'.\n");
            i -= 1;
            j -= 1;
            G[i].emplace_back(j, v);
            if (i != j)
                G[j].emplace_back(i, v);
        }
    }

    if (num_entries_readed != expected_num_lines)
        throw runtime_error(string("expected to read ") + to_string(expected_num_lines) + "' entries but only read " + to_string(num_entries_readed) + "lines. The file '" + file_name + "' could be truncated or damaged.\n");

    fp.close();

    // matrix to csr
    for (auto it = G.begin(); it != G.end(); ++it)
        sort((it->second).begin(), (it->second).end());

    for (int r = 0; r != num_rows_; r++)
    {
        ia_.push_back((signed)a_.size());
        auto const Gr = G.find(r);
        if (Gr == G.end())
            continue;
        for (auto const &p : Gr->second)
        {
            ja_.push_back(p.first);
            a_.push_back(p.second);
        }
    }
    ia_.push_back((signed)a_.size());
}

// ============================================================================
// back multiply a vector, Y=A*X
// ============================================================================
void MtxSp::MultiplyVector(int nrows, double *x, double *y)
{
    for (int i = 0, iEnd = nrows; i < iEnd; i++)
    {
        double t = 0;
        for (int jt = ia_[i], jtEnd = ia_[i + 1]; jt < jtEnd; jt++)
            t += a_[jt] * x[ja_[jt]];
        y[i] = t;
    }
} //end of fun MultiplyVector

// ============================================================================
// back multiply a dense matrix, Y=A*X
// ----------------------------------------------------------------------------
// X is a nrows by K matrix
// Y[i][k] = A[i][j] * X[j][k]
// X is stored in row domain! for(i, k, j)
//   if X is not stored in row domain but col domain,
//   better to use for(k,i,j) or for(i,j,k)
// ============================================================================
void MtxSp::MultiplyMatrix(int nrows, int K, double *x, double *y)
{
#ifndef DENSE_MATRIX_COL_DOMAIN
    for (int i = 0, iEnd = nrows; i < iEnd; i++)
    {
        const int itK = i * K;
        for (int k = 0; k < K; k++)
            y[itK + k] = 0;
        for (int jt = ia_[i], jtEnd = ia_[i + 1]; jt < jtEnd; jt++)
        {
            const int j = ja_[jt];
            const int jtK = j * K;
            const double a = a_[jt];
            for (int k = 0; k < K; k++)
            {
                y[itK + k] += a * x[jtK + k];
            }
        }
    }
#else
    for (int i = 0, iEnd = nrows; i < iEnd; i++)
    {
        const int itK = i * K;
        for (int k = 0; k < K; k++)
        {
            double t = 0;
            for (int jt = ia_[i], jtEnd = ia_[i + 1]; jt < jtEnd; jt++)
            {
                t += a_[jt] * xglb[ja_[jt] * K + k];
            }
            yloc[itK + k] = t;
        }
    }
#endif
} //end of fn MultiplyMatrix


// dump to screen
void MtxSp::dump(){
    cout << toString();
}


// ============================================================================
// dump to string
// ============================================================================
string MtxSp::toString()
{
    stringstream ss;

    ss << "\n### MtxSp class info \n"
       << "{ num_rows_ : " << num_rows_ << ", num_cols_ : " << num_cols_
       << ", ia_[" << ia_.size() << "]{...}, ja_[" << ja_.size() << "]{...}, a_[" << a_.size() << "]{...} }\n";

    auto dumpVec = [](auto const &vec) 
    {
        stringstream ss;
        int const NumMaxDisp = 50;
        size_t i = 0;
        for (i = 0; i < vec.size() && i < NumMaxDisp; i++)
            ss << vec[i] << ",";
        if (i >= NumMaxDisp)
            ss << " ...\n";
        else
            ss << "\n";
        return ss.str();
    };

    ss << "ia_[" << ia_.size() << "] :" << dumpVec(ia_);
    ss << "ja_[" << ja_.size() << "] :" << dumpVec(ja_);
    ss << "a_ [" << a_.size() << "] :" << dumpVec(a_);
    ss << "\n";
    return ss.str();
}
