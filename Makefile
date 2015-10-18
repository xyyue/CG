ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=0                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0	  # Use the shared low level
USE_CUDA=0		  # to activate/deactivate cuda
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE		:= main 
# List all the application source files here
GEN_SRC		:= main.cc mmio.cc# .cc files
GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= 
CC_FLAGS	:= -DMAX_FIELDS=64 -DFIELD_LOG2=6 #-DLEGION_PROF -DPREDICATED_EXECUTION
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:=

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

# All these variables will be filled in by the runtime makefile
LOW_RUNTIME_SRC	:=
HIGH_RUNTIME_SRC:=
GPU_RUNTIME_SRC	:=
MAPPER_SRC	:=

include $(LG_RT_DIR)/runtime.mk
#
## General shell commands
#SHELL	:= /bin/sh
#SH	:= sh
#RM	:= rm -f
#LS	:= ls
#MKDIR	:= mkdir
#MV	:= mv
#CP	:= cp
#SED	:= sed
#ECHO	:= echo
#TOUCH	:= touch
#MAKE	:= make
#ifndef GCC
#GCC	:= g++
#endif
#ifndef NVCC
#NVCC	:= $(CUDA)/bin/nvcc
#endif
#SSH	:= ssh
#SCP	:= scp
#
#common_all : all
#
#.PHONY	: common_all
#
#GEN_OBJS	:= $(GEN_SRC:.cc=.o)
#LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
#HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
#MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)
## Only compile the gpu objects if we need to 
#ifndef SHARED_LOWLEVEL
#GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
#GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
#else
#GEN_GPU_OBJS	:=
#GPU_RUNTIME_OBJS:=
#endif
#
#ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS)
#
#all:
#	$(MAKE) $(OUTFILE)
#
## If we're using the general low-level runtime we have to link with nvcc
#$(OUTFILE) : $(ALL_OBJS)
#	@echo "---> Linking objects into one binary: $(OUTFILE)"
#ifdef SHARED_LOWLEVEL
#	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)
#else
#	$(NVCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)
#endif
#
#$(GEN_OBJS) : %.o : %.cc
#	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
#
#$(LOW_RUNTIME_OBJS) : %.o : %.cc
#	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
#
#$(HIGH_RUNTIME_OBJS) : %.o : %.cc
#	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
#
#$(MAPPER_OBJS) : %.o : %.cc
#	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
#
#$(GEN_GPU_OBJS) : %.o : %.cu
#	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)
#
#$(GPU_RUNTIME_OBJS): %.o : %.cu
#	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)
#
#clean:
#	@$(RM) -rf $(GEN_OBJS) $(OUTFILE)
#
#cleanall:
#	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)