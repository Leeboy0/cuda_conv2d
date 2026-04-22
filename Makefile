NVCC     := nvcc
NVCCFLAGS := -O3 -arch=sm_120 -std=c++17 -Xcompiler -Wall
# Change sm_86 to match your GPU:
#   sm_75  -> Turing  (RTX 20xx)
#   sm_86  -> Ampere  (RTX 30xx)
#   sm_89  -> Ada     (RTX 40xx)
# 	sm_120 -> Blackwell (RTX50xx)
TARGET   := benchmark
SRCS := benchmark.cu conv2d_naive.cu

all: $(TARGET)

$(TARGET): $(SRCS) conv2d.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: all clean