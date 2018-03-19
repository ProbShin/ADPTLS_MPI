# source files
CPPSRC = $(wildcard src/*.cpp)
CPPOBJ = $(CPPSRC:%.cpp=%.o)
CSRC = $(wildcard *.c)
COBJ = $(CSRC:%.c=%.o)
SRC = $(CPPSRC) $(CSRC)
OBJ = $(CPPOBJ) $(COBJ)
EXE = adptls.out

# compiler
#COMPILE = g++      # gnu
#COMPILE = icpc     # intel
COMPILE = mpiicpc # intel mpi

# compile flags
CCFLAGS = -Wall -DMKL_ILP64 -qopenmp -O3 -std=c++11 
#-D DEBUG_ADPTLS_CPP
# -D DEBUG_SHOW_DETAILS

# link flags
LDFLAGS = -Wall 

# MKL library
MKL_MIC_ENABLE = 1
MKL =    ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -liomp5 -lpthread -lm


#FLAGS= -DMKL_ILP64 -qopenmp -std=c++11 -I${MKLROOT}/include 

INCLUDES = -I. -Iinc   -I${MKLROOT}/include 



all: $(EXE)

%.o : %.cpp
	$(COMPILE) $(INCLUDES) $(CCFLAGS) -c $< -o $@

$(EXE): $(OBJ)
	$(COMPILE) $^ $(LDFLAGS) $(MKL)  -o $@

clean:
	rm -f $(OBJ) $(EXE)

