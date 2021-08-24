cmake_minimum_required(VERSION 3.10)

# set the project name
project(ADPTLS VERSION 1.0)

# add the executable
add_executable(testall2all.out 
    app/test_mpi_all2all/main.cpp 
    src/MtxDen.cpp
    src/MtxSp.cpp
    src/MtxSpMpi.cpp
    src/MtxSpMpiA2a.cpp
    src/FtlMtxMpi.cpp
    src/adptls.cpp
    )

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)
target_include_directories(testall2all.out PRIVATE 
                        ${MPI_CXX_INCLUDE_PATH} 
                        ${PROJECT_SOURCE_DIR}/inc
                        )
target_link_libraries(testall2all.out PUBLIC MPI::MPI_CXX)


