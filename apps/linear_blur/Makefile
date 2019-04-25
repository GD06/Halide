include ../support/Makefile.inc 

BIN ?= bin 

all: 

$(BIN)/linear_blur.generator: linear_blur_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/linear_blur.a: $(BIN)/linear_blur.generator
	@-mkdir -p $(BIN)
	$^ -g linear_blur -o $(BIN) -f linear_blur target=$(HL_TARGET) auto_schedule=true

$(BIN)/linear_blur_gpu.a: $(BIN)/linear_blur.generator 
	@-mkdir -p $(BIN)
	$^ -g linear_blur -o $(BIN) -f linear_blur_gpu target=cuda auto_schedule=false  

$(BIN)/run_linear_blur: run_linear_blur.cpp $(BIN)/linear_blur.a $(BIN)/linear_blur_gpu.a
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)

