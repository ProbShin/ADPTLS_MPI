#ifndef MTXMKT_HPP
#define MTXMKT_HPP

// ======================================================================
// > File Name: MtxMkt.hpp
// > Author: xin cheng
// > Descriptions: wrape of read file of matrix market file format
// > Last Updated Time: 2021 Aug
// > Version: v0001 
// ======================================================================

#include <string>
#include <vector>

using std::string;
using std::vector;


class MtxMkt {
public:
    MtxMkt() {};
    virtual ~MtxMkt(){};

public:
    virtual static read_matrix(string const &file_name, vector<int> &ai, vector<int> &aj, vector<double> &a);
    virtual static read_matrix(string const &file_name, vector<int> &ai, vector<int> &aj, vector<double> &a, int& num_rows, int& num_cols);
    virtual static read_matrix_only_struct(string const &file_name, vector<int> &ai, vector<int> &aj, vector<double> &a);
    virtual static read_matrix_only_struct(string const &file_name, vector<int> &ai, vector<int> &aj, vector<double> &a);

    virtual static read_matrix_to_dense_array(string const &file_name, vector<int> &a, int& num_rows, int& num_cols);
    virtual static read_mpi_local_matrix(string const &file_name, int rank, int nproc, vector<int> &ai, vector<int> &aj, vector<double> &a, int& num_rows_local, int& num_cols_local, int& num_rows_glb_, int& num_cols_glb_);

    virtual static check


    

};

#endif