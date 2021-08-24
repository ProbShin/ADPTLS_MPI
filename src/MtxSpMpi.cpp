/************************************************************************
  > File Name: MtxSpMpi.cpp
  > Author: xin cheng
  > Descriptions: implementation of class mtxSpMpi
  > Created Time: Before 2020
 ************************************************************************/
#include "MtxSpMpi.hpp"
#include <unordered_map>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>             // regex_replace

using std::cerr;
using std::ifstream;
using std::pair;
using std::runtime_error;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::vector;
using std::regex_replace;
using std::to_string;

// read a matrix market file 
void MtxSpMpi::do_read_matrixmarket(const string &file_name)
{
    auto & ia = get_ia();
  auto & ja = get_ja();
  auto &  a = get_a();

  ia.clear(); ia.shrink_to_fit();
  ja.clear(); ja.shrink_to_fit();
  a .clear(); a .shrink_to_fit();

  row_rcvcnt_.clear(); row_rcvcnt_.shrink_to_fit();
  row_displs_.clear(); row_displs_.shrink_to_fit();
  
  num_rows_loc_ = num_cols_loc_ = 0;
  num_rows_glb_ = num_cols_glb_ = 0;

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
        cerr << "Warning! MatrixMarket file '" << file_name << " is a integer matrix, "
                                                               "the integer data will be read as double float.\n";
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

  // calculate globle and local dimension information
  num_rows_glb_ = r;
  num_cols_glb_ = c;
  num_cols_loc_ = c;

  row_rcvcnt_.clear();
  row_rcvcnt_.resize(nproc_, num_rows_glb_ / nproc_);
  row_displs_.clear();
  row_displs_.resize(nproc_ + 1, 0);

  for (int i = 0, iEnd = num_rows_glb_ % nproc_; i < iEnd; i++)
    row_rcvcnt_[i]++;
  for (int i = 1; i <= nproc_; i++)
    row_displs_[i] = row_displs_[i - 1] + row_rcvcnt_[i - 1];

  num_rows_loc_ = row_rcvcnt_[rank_];

  // read the matrix
  int num_entries_readed = 0;
  unordered_map<int, vector<pair<int, double>>> G;
  int i, j;
  int iBeg = row_displs_[rank_], iEnd = row_displs_[rank_ + 1];
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
      if (iBeg <= i && i < iEnd)
        G[i].emplace_back(j, v);
      if (++i >= num_rows_glb_)
      {
        i = 0;
        j += 1;
      }
    }
  }
  else if (!bSymm)
  { // matrix is non-symm
    while (getline(fp, line) && num_entries_readed <= expected_num_lines)
    {
      if (line.empty() || line[0] == '%')
        continue;
      num_entries_readed += 1;
      ss.str(line);
      ss >> i >> j >> v;
      i -= 1;
      j -= 1;
      if (iBeg <= i && i < iEnd)
        G[i].emplace_back(j, v);
    }
  }
  else
  { // matrix is symmetric
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
      if (iBeg <= i && i < iEnd)
        G[i].emplace_back(j, v);
      if (i != j && i > iBeg && j >= iBeg && j < iEnd)
        G[j].emplace_back(i, v);
    }
  }

  if (num_entries_readed != expected_num_lines)
    throw runtime_error(string("expected to read ") + to_string(expected_num_lines) + "' entries but only read " + to_string(num_entries_readed) + "lines. The file '" + file_name + "' could be truncated or damaged.\n");

  fp.close();

  // [optional] make sure ordered per row
  for (auto it = G.begin(); it != G.end(); ++it)
    sort((it->second).begin(), (it->second).end());

  // G into CSR local
  for (i = iBeg; i != iEnd; i++)
  {
    ia.push_back((signed)a.size());
    auto const gi = G.find(i);
    if (gi != G.end())
    {
      for (auto const &p : gi->second)
      {
        ja.push_back(p.first);
        a.push_back(p.second);
      }
    }
  }
  ia.push_back((signed)a.size());
}



// ============================================================================
// back multiply a vector xglb into yglb.  Yglb <- MPI_Allgatherv( Aloc*xglb)
// ============================================================================
void MtxSpMpi::MultiplyVector_Allgatherv(int nrows_loc,        double *xglb, double *yloc, double*yglb, int*recvcounts, int*displs){
  MtxSp::MultiplyVector(nrows_loc,    xglb, yloc);
  MPI_Allgatherv(yloc, nrows_loc, MPI_DOUBLE, yglb, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD); 
} 


// ============================================================================
// back multiply a dense matrix locally, yloc <- allgatherv( Aloc*Xglb )
// ============================================================================
void MtxSpMpi::MultiplyMatrix_Allgatherv(int nrows_loc, int K, double *xglb, double* yloc, double*yglb, int*recvcounts, int*displs){
  MtxSp::MultiplyMatrix(nrows_loc, K, xglb, yloc);
  MPI_Allgatherv(yloc, nrows_loc*K, MPI_DOUBLE, yglb, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD); 
}


// toString
string MtxSpMpi::toString(){
    auto dumpVec = [](auto const &vec)
    {
        stringstream ss;
        int const NumMaxDisp = 50;
        size_t i = 0;
        for (i = 0; i < vec.size() && i < NumMaxDisp; i++)
            ss << vec[i] << ",";
        if (i >= NumMaxDisp)
            ss << " ... ";
        else
            ss << " ";
        return ss.str();
    };

    stringstream ss;
    if (rank_ == 0)
    {
        ss << "\n" << regex_replace(MtxSp::toString(), std::regex("\n"), "\n> ");
        ss << "### MtxSpMpi with nproc_:" << nproc_ << " rank_/nproc_ { num_rows_loc_/num_rows_glb_, num_cols_loc_/num_cols_glb_, row_rcvcnt_[]{...} row_displs[]{...} }\n";
    }
 
  ss << "rank : "<<rank_<<"/"<<nproc_ << " {";
  ss << num_rows_loc_ << "/" << num_rows_glb_ << ", ";
  ss << num_cols_loc_ << "/" << num_cols_glb_ << ", ";
  ss << " row_rcvcnt["<<row_rcvcnt_.size()<<"]{"<< dumpVec(row_rcvcnt_)<<"}";
  ss << " row_displs["<<row_displs_.size()<<"]{"<< dumpVec(row_displs_)<<"}";
  ss << " a_[" << a_.size() << "]{" << dumpVec(a_) << "}";
  ss << " ia_[" << a_.size() << "]{" << dumpVec(a_) << "}";
  ss << " ja_[" << a_.size() << "]{" << dumpVec(a_) << "}";
  ss << "}\n";
  return ss.str();
}


// ============================================================================
// dump to screen
// ============================================================================
void MtxSpMpi::dump(){
  MPI_Barrier(MPI_COMM_WORLD);
  int msg=0;
  if(rank_!=0) MPI_Recv(&msg, 1, MPI_INT, rank_-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  cerr << toString();

  if(rank_+1!=nproc_) MPI_Send(&msg, 1, MPI_INT, rank_+1, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}


// dump to screen
void MtxSpMpi::dump(vector<int> &v, string const &vname) const
{
  MPI_Barrier(MPI_COMM_WORLD);
  int msg = 0;
  if (rank_ != 0)
    MPI_Recv(&msg, 1, MPI_INT, rank_ - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  int const MaxNumElmToShow = 100;

  stringstream ss;
  ss << "\nrank: " << rank_ << "\n"
     << (vname.empty() ? "vec" : vname) << "(" << v.size() << "):[";
  for(auto i=0; i<v.size() && i<MaxNumElmToShow; i++) ss << v[i] << ",";
  ss << (v.size() > MaxNumElmToShow ? "...]" : "]") << "\n";
  cerr << ss.str() << std::flush;

  if (rank_ + 1 != nproc_)
    MPI_Send(&msg, 1, MPI_INT, rank_ + 1, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}
