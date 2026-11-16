PYTHON ?= python3
CXX = g++
SITE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('purelib'))")
CXXFLAGS = -O3 -std=c++17 -fPIC -shared
INCLUDES = -Iinclude -I$(SITE)/nvidia/cuda_runtime/include -I$(SITE)/nvidia/cuda_nvcc/include

.PHONY: all
all: native64.so native128.so

native%.so: native.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) -DBENCH_BATCH=$* $< $(INCLUDES) -ldl -pthread -o $@
