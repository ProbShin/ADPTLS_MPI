/************************************************************************
  > File Name: src/mtxSpMpiA2a.cpp
  > Author: xin cheng
  > Descriptions: implementation of class mtxSpMpiA2a
  > Created Time: Aug 1 2021
 ************************************************************************/
#include "mtxSpMpiA2a.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cassert>

using std::cerr;
using std::string;
using std::getline;
using std::vector;
using std::ifstream;
using std::stringstream;
using std::runtime_error;
using std::unordered_map;
using std::to_string;
using std::unordered_set;

void MtxSpMpiA2a::doReadMtxSpecial4MpiA2a(const string& filename) {
    auto& ia = get_ia();
    auto& ja = get_ja();
    auto& a  = get_a();
    
    ia.clear(); ia.shrink_to_fit();
    ja.clear(); ja.shrink_to_fit();
    a.clear();   a.shrink_to_fit();

    row_rcvcnt_.clear(); row_rcvcnt_.shrink_to_fit();
    row_displs_.clear(); row_displs_.shrink_to_fit();
    
    send_buf_.clear();           send_buf_.shrink_to_fit();
    send_buf_map_i2nzc_.clear(); send_buf_map_i2nzc_.shrink_to_fit(); 
    send_counts_.clear();        send_counts_.shrink_to_fit();
    send_displs_.clear();        send_displs_.shrink_to_fit();

    recv_buf_.clear();             recv_buf_.shrink_to_fit();
    recv_buf_map_j2index_.clear(); recv_buf_map_j2index_.shrink_to_fit();
    recv_counts_.clear();          recv_counts_.shrink_to_fit();
    recv_displs_.clear();          recv_displs_.shrink_to_fit();


    bool bDens = false;
    bool bSymm = false;

    if(file_name.empty()) throw runtime_error("file name is empty.\n");
    ifstream fp(file_name);
    if(!fp.is_open()) throw runtime_error(string("cannot open file'") + file_name +"'.\n"); 
    string line, word;
    stringstream ss;
    getline(fp, line);
    ss.str(line);
    {   // check the format
        string word1, word2, word3;
        if( !(ss>>word) || (word!="%%MatrixMarket") || !(ss>>word) || (word!="matrix")
            || (ss>>word1>>word2>>word3) ) throw runtime_error(
        string("MatrixMarket file '") + filename + "' format error with headline '" + line + "'\n");
        if( word1 == "array" ) {
            bDense = true;
            cerr<<"Warning! MatrixMarket file '"<<filename<<"' contains a dense matrix while expect a sparse one.\n");
        }
        else if( word1 != "coordinate" ) throw runtime_error(string("MatrixMarket file '") + filename 
                + "' is neigher 'array' nor 'coordinate', with head line '" + line + "'\n" );
        if( word2 != "real" ) {
            if (word2 == "integer") cerr<< "Warning! MatrixMarket file '"<<filename<<" is a integer matrix, " \ 
                "the integer data will be read as double float.\n";
            else throw runtime_error( string("The matrix of MatrixMarket file '") + filename + "' is of type '" + word2
                + "' which is not supported. The only supported format is 'real' or 'integer'.\n");
        }
        if( word3 != "general" ) {
            bSymm = true;
            if(word3 != "symmetric"){
                if (word3 == "skew-symmetric" || word3 == "Hermitian") cerr << "Warning! The '" << word3 \
                    << "'matrix of MatrixMarket file '"<< filename << "' will be read as the general symmetric graph.\n";
                else throw runtime_error( string("The '") + word3 + "' matrix of file '" + filename + "' is not supported.\n");
            }
        }
    }

    while(getline(fp,line)){
        if(line.empty() || line[0]=='%') continue; 
        break;
    }

    int r, c, expected_num_lines=0;
    ss.clear();
    ss.str(line)
    if( !(ss>>r>>c) || (!bDens && !(ss>>expected_num_lines)) ) throw runtime_error( string( "could not read the matrx " \
        "dimension from the line '") + line + "' of matrix file '" + filename + "'");
    if(bDens) expected_num_lines = r*c;
    if(bSymm && r!=c) throw runtime_error(string("symmetric matrix '") + filename + "' number_of_rows (" + to_string(r) 
        + ") != number_of_cols (" + to_string(c)+").\n");

    // calculate dimension information
    num_rows_glb_ = r;
    num_cols_glb_ = c;
    num_cols_loc_ = c;

    row_rcvcnt_.clear(); row_rcvcnt_.resize(nproc_, num_rows_glb_/nproc_);
    row_displs_.clear(); row_displs_.resize(nproc_+1, 0);

    for(int i=0, iEnd=num_rows_glb_%nproc_; i<iEnd; i++) row_rcvcnt_[i]++;
    for(int i=1; i<=nproc_; i++) row_displs_[i] = row_displs_[i-1] + row_rcvcnt_[i-1];
  
    num_rows_loc_ = row_rcvcnt_[rank_];

    // read the matrix
    int num_entries_readed = 0;
    unordered_map<int, vector<pair<int, double>>> G;
    int i,j;
    int iBeg = row_displs_[rank_], iEnd = row_displs_[rank_+1];
    double v;
    ss.clear();
    i=0,j=0,v=.0;
    if(bDens){
        while(getline(fp,line) && num_entries_readed <= expected_num_lines ) {
            if(line.empty() || line[0]=='%') continue;
            num_entries_readed += 1;
            ss.str(line);
            ss>>v;
            if(iBeg<=i && i<iEnd) G[i].emplace_back(j,v);
            if(++i>=num_rows_glb) {
                i=0;
                j+=1;
            }     
        }
    }
    else if(!bSymm){
        while(getline(fp, line) && num_entries_readed <= expected_num_lines ) {
            if(line.empty() || line[0]=='%') continue;
            num_entry_readed += 1;
            ss.str(line)
            ss>>i>>j>>v;
            i-=1;
            j-=1;
            if(iBeg<=i && i<iEnd) G[i].emplace_back(j,v);
        }
    }
    else{ // matrix is symmetric
        while(getline(fp, line) && num_entries_readed <= expected_num_lines ) {
            if(line.empty() || line[0]=='%') continue;
            num_entry_readed += 1;
            ss.str(line)
            ss>>i>>j>>v;
            if(i<j) throw runtime_error(string("read a upper diagonal entry '")+ to_string(i) + ", " 
                + to_string(j) + ", " + to_string(v) + "' around " + to_string(num_entries_readed) 
                + "' lines of file '" + filename + "'.\n"); 
            i-=1;
            j-=1;
            if(iBeg<=i && i<iEnd) G[i].emplace_back(j,v);
            if(i!=j && i>iBeg && j>=iBeg && j<iEnd) G[j].emplace_back(i, v);
        }
    }

    if(num_entries_readed != expected_num_lines) throw runtime_error( string("expected to read " 
            + to_string(expected_num_lines) + "' entries but only read "+ to_string(num_entries_readed) 
            + "lines. The file '" + filename + "' could be truncated or damaged.\n");

    fp.close();

    // [optional] make sure ordered per row
    for(auto it = G.begin(); it!=G.end(); ++it)
        sort((it->second).begin(), (it->second).end());

    // G into CSR local
    for(i=iBeg; i!=iEnd; i++) {
        ia.push_back((signed)a.size());
        auto gi = G.find(i);
        if( (auto gi = G.find(i) != G.end() ){
            for(auto&p : *gi){
                ja.push_back(p.first);
                 a.push_back(p.second);
            }
        }
    } 
    ia.push_back((signed)a.size());

    // pre-process for all2all communication
    vector<int>& nzc = recv_buf_;
    {
        unordered_set<int> s_nzc;
        for(auto const& p : G)
            for(auto const & j : p.second)
                s_nzc.insert(j);
        nzc.assign(s_nzc.begin(), s_nzc.end());
        sort(nzc.begin(), nzc.end());
    }

    vector<int>& nzc_counts = recv_counts_;
    vector<int>& nzc_displs = recv_displs_;
    
    nzc_counts_.assign(nproc_, 0);
    nzc_displs_.assign(nproc_+1, 0);
    vector<int> col_counts(nproc_, num_cols_glb_/nproc_);
    vector<int> col_displs(nproc_+1, 0);
    for(int i=0, iEnd=num_cols_glb_%nproc_; i<iEnd; i++) col_counts[i]++;
    for(int i=1; i<=nproc_; i++) col_displs[i] = col_displs[i-1] + col_counts[i-1];

    int iBeg = 1, cnt = 0;
    for(auto const cid : nzc) { 
        for(int i = iBeg; i<=nproc_; i++){
            if( cid < col_disls[i]){
                nzc_counts[i-1] += 1;
                iBeg = i;
                break;
            }
        }
    }
    for(int i=1; i<=nproc_; i++) nzc_displs[i] = nzc_displs[i-1] + nzc_counts[i-1];
    assert(nzc_displs[nproc_] == num_cols_glb_);


    send_counts_.assign(nproc_, 0);
    MPI_Alltoall(&nzc_counts[0], 1, MPI_INT, &send_counts_[0], 1, MPI_INT, MPI_COMM_WORLD);
    
    send_displs_.assign(nproc_+1, 0);
    for(int i=1; i<=nproc; i++) send_displs_[i] = send_displs_[i-1] + send_counts_[i-1];
    
    send_buf_.assign(send_displs_[nproc_], 0);
    send_buf_map_i2nzc_.assign(send_displs_[nproc], 0);
    MPI_Alltoallv( &nzc[0], &nzc_counts[0], &nzc_disps[0], MPI_INT, 
                &send_buf_map_i2nzc_[0], &send_counts_[0], &send_displs[0], MPI_INT, MPI_COMM_WORLD);
    
    for(auto & x : send_buf_map_i2nzc_)  // localize
        x -= col_displs[rank_];

    recv_buf_map_j2index_.assign(num_cols_glb_, 0);   // r_b_m_j[j] ->  index_of_value_from_recv_buf
    {   
        int i = 0;
        for(auto const x : nzc)
            recv_buf_map_j2index_[x] = i++; 
    }
    
}


// MatVec multiplication of local matrix and local vector
// Matrix Multiply Vector. y = A * x 
// @param[in]  xloc is an double array of x. It must hold at least rows_loc() elements.
// @param[out] yloc is an double array of y. It should be able to hold at least rows_loc() elements.
void MtxSpMpiA2a::MultiplyVector_all2allv(double* xloc, double *yloc){
    {   // fill in the sendbuf
        auto i=0;
        for(auto nzc : send_buf_map_i2nzc) send_buf[i++] = xloc[nzc];
    }
    MPI_Alltoallv(&send_buf[0], &send_counts[0],
                &send_displs[0], MPI_INT, &recv_buf[0],
                &recv_counts[0], &recv_displs[0], MPI_INT, MPI_COMM_WORLD);
    {   // MatVec
        auto const & a  = get_a();
        auto const & ia = get_ia();
        auto const & ja = get_ja();
        auto const M = rows_loc();
        auto const N = cols_loc();
    
        for(auto i=0; i<M; i++){
            yloc[i] = 0;
            for(auto jt=ia[i],jtEnd=ia[i+1]; jt<jtEnd; jt++){
                yloc[i] +=  a[jt] * recv_buf[ recv_buf_map_j2index[ja[jt]] ];
            }
        }
    }
}