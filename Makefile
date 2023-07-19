#                         CUFIM-NVLINK - MAKEFILE
# Structure ===================================================================
# 
#        main
# 
# Switches ====================================================================
#include cfg/switches.def
# pass nt , $(nproc)
TW_FLAGS= -DTRACY_ENABLE#\
-D TW_OMP_EN \
-D TW_OMP_NT=4 \
-D TW_CUDA_EN 
# cpp version, cuda version, packages, ..
# TW_PAR_MAKE
#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#define NK_IMPLEMENTATION
# print switches
# Paths =======================================================================
SRC_DIR=src
LIB_DIR=lib
OBJ_DIR=obj
DIRS= $(SRC_DIR)/core $(SRC_DIR)/io $(SRC_DIR)/eikonal $(SRC_DIR)

# $(foreach var, list, text)
SRC_FILES= $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
# $(patsubst pattern,replacement,text)
OBJ_FILES= $(patsubst src/%.cpp, obj/%.o, $(SRC_FILES))

SRC_CUDA= $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cu))
OBJ_CUDA= $(patsubst src/%.cu, obj/%.o, $(SRC_CUDA))

# CUDA
CUDA_ROOT_DIR=/usr/local/cuda
#CUDA_ROOT_DIR=/usr/include/cuda
#CUDA_ROOT_DIR=/usr/local/cuda-12.1
#CUDA_SAMPLES_ROOT_DIR=/home/tom/examples/cuda-samples/Common

CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
#CUDA_SAMPLES_INC_DIR= -I$(CUDA_SAMPLES_ROOT_DIR)
CUDA_LINK_LIBS= -lcudart -lcublas -lcudadevrt
# -rdc=true

# Compiler options ============================================================
# CXX compiler options:
CXX=g++
CXX_OPTS= -std=c++17 -Wfatal-errors -Wno-unused-result -Wall -Wextra -MMD $(TW_FLAGS) 
CXX_LIBS= -I/$(LIB_DIR) 
#-lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

#-g3
#ifdef TW_DEBUG
CXX_OPTS+= -DDEBUG -g -fopenmp
# -O3
#else
#CXX_OPTS+=-O3
#endif

#VULKAN_SDK_PATH = /home/user/VulkanSDK/x.x.x.x/x86_64
#STB_INCLUDE_PATH= /home/user/libraries/stb
#TINYOBJ_INCLUDE_PATH= /home/user/libraries/tinyobjloader
#CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include -I$(STB_INCLUDE_PATH) -I$(TINYOBJ_INCLUDE_PATH)

# any non-system header -> -MMD, system headers -> -MD, dummy rules if headers are removed -> -MP
# optimization flags

# NVCC compiler options:
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
#ARCH= sm_70
#-arch=compute_60 -code=sm_70 -arch=$(ARCH)
#--gpu-architecture=compute_50 --gpu-code=compute_50,sm_50,sm_53
#ampere:
#compute_80, compute_86 and compute_87
#sm_80, sm_86 and sm_87
# todo: automatically determine sm

NVCC= nvcc 
NVCC_OPTS= -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -g -arch=sm_80
# -arch=sm_86
# -use_fast_math
# -arch=sm_50
#-Xptxas -v -O{1|2|3}
#,-abi=no
#-rdc=true
#--gpu-architecture=sm_80 
#-arch=sm_86
NVCC_LIBS= $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
#$(CUDA_SAMPLES_INC_DIR)
#CUDA_MODULE_LOADING=LAZY

#CFLAGS = -c -m32 -I$(CUDAPATH)/include
#NVCCFLAGS = -c -I$(CUDAPATH)/include

# Formatting ==================================================================
NO_COL="\033[0m"
RD_COL="\033[1;31m"
GR_COL="\033[1;32m"
BL_COL="\033[1;34m"
CY_COL="\033[1;36m"
YL_COL="\033[1;33m"

# Targets =====================================================================
#obj -> bin?
# top down?
# target: prerequisites
# $@ Outputs target
# $? Outputs all prerequisites newer than the target
# $^ Outputs all prerequisites
# $< ???
# source files --(compile)-> obj_files --(link)-> exe_file
default: cufim-nvlink-par
#cufim-nvlink
#	make -j$(nproc) cufim-nvlink
# -j for parallel
#compute-sanitizer ./cufim-nvlink
cufim-nvlink-par: 
	make -j8 cufim-nvlink

#tracy? -> make release build

# Build step for C++ sources:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo $(GR_COL)"* compiling:"$(BL_COL) $< $(GR_COL)"to"$(YL_COL) $@ $(NO_COL)
	@mkdir -p $(@D)
	$(CXX) -c $< -o $@ $(CXX_OPTS) $(CUDA_INC_DIR)
#$(CUDA_SAMPLES_INC_DIR)

# Build step for CUDA sources:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	@echo $(GR_COL)"* compiling:"$(BL_COL) $< $(GR_COL)"to"$(YL_COL) $@ $(NO_COL)
	@mkdir -p $(@D)
	$(NVCC) -c $< -o $@ $(NVCC_OPTS) $(CUDA_INC_DIR)
#$(CUDA_SAMPLES_INC_DIR)

cufim-nvlink: $(OBJ_FILES) $(OBJ_CUDA)
	@echo $(GR_COL)"* linking:"$(YL_COL) $@ $(NO_COL)
	$(CXX) $^ -o $@ $(CXX_OPTS) $(CXX_LIBS) $(NVCC_LIBS)

# mark clean as not associated with files
.PHONY: clean

# run unit tests (compile in debug?)
test: clean cufim-nvlink-par
	./cufim-nvlink --test

heart: clean cufim-nvlink-par
	./cufim-nvlink --mesh=data/heart/S62.vtk --part=data/heart/S62.7560.tags --plan=data/heart/S62.plan.json --odir=data/output/cufim-heart-out.vtk --reps=1

memtest: clean cufim-nvlink-par
	valgrind ./cufim-nvlink

cumemtest: clean cufim-nvlink-par
	cuda-memcheck --leak-check full ./cufim-nvlink

# clean obj files and executable
clean:
	@echo $(GR_COL)"* cleaning"$(NO_COL)
	@rm -rfv $(OBJ_DIR)
	@rm -fv cufim-nvlink

# Dependencies
-include $(OBJ_FILES:.o=.d) $(OBJ_CUDA:.o=.d)
# for what?