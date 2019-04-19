# $File: Makefile
# $Date: Thu Aug 03 17:42:32 2017 -0700

OBJ_DIR = obj
TARGET = image-stitching

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXX ?= g++
	EIGEN3_INCLUDE_DIR ?= /usr/include/eigen3
endif
ifeq ($(UNAME_S),Darwin)
	CXX ?= g++
	EIGEN3_INCLUDE_DIR ?= /usr/local/include/eigen3
endif
INCLUDE_DIR = -I. -isystem $(EIGEN3_INCLUDE_DIR) 


OPTFLAGS ?= -O3 -msse3 -march=native
#OPTFLAGS ?= -g3 -fsanitize=address,undefined -O2 -lasan
#OPTFLAGS ?= -g3 -fsanitize=leak -O2 -lubsan
#DEFINES = -DDEBUG  	# comment out this line improves speed

# optional extra packages
#LIBS = opencv
#INCLUDE_DIR += $(shell pkg-config --cflags $(LIBS))
#LDFLAGS += $(shell pkg-config $(LIBS) --libs)

# use mkl for eigen
ifdef MKLROOT
	INCLUDE_DIR += -I$(MKLROOT)/include
	LDFLAGS += -m64 -L$(MKLROOT)/lib/intel64 -L$(MKLROOT)/../compiler/lib/intel64
	LDFLAGS += -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
	DEFINES += -DEIGEN_USE_MKL_ALL
endif

CXXFLAGS ?=
CXXFLAGS += $(INCLUDE_DIR) -Wall -Wextra -Wnon-virtual-dtor
CXXFLAGS += $(DEFINES) -std=c++11 $(OPTFLAGS)

LDFLAGS += $(OPTFLAGS)
LDFLAGS += -ljpeg

SHELL = bash
# sources to include
SRCDIRS = . 
ccSOURCES = $(shell find $(SRCDIRS) -name "*.C" | sed 's/^\.\///g')
OBJS = $(addprefix $(OBJ_DIR)/,$(ccSOURCES:.C=.o))
DEPFILES = $(OBJS:.o=.d)

.PHONY: all clean run

all: $(TARGET)

ifneq ($(MAKECMDGOALS), clean)
sinclude $(DEPFILES)
endif

$(OBJ_DIR)/%.o: %.C
	@echo "[C] $< ..."
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJ_DIR)/%.d: %.C Makefile
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.C=.o) $(OBJ_DIR)/$(<:.C=.d)" "$<"  > "$@"

$(TARGET): ImageStitcherTest.C $(OBJS)
	@echo "Linking ..."
	@$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)
	@echo "done."

clean:
	@rm -rvf $(OBJ_DIR)

