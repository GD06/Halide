#include "Halide.h"

#include "daubechies_constants.h"

namespace {

Halide::Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");

class haar_x : public Halide::Generator<haar_x> {
public:
    Input<Buffer<float>> in_{"in" , 2};
    Output<Buffer<float>> out_{"out" , 3};

    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 8};

    void generate() {
        Func in = Halide::BoundaryConditions::repeat_edge(in_);

        out_(x, y, c) = select(c == 0,
                              (in(2*x, y) + in(2*x+1, y)),
                              (in(2*x, y) - in(2*x+1, y)))/2;
        out_.unroll(c, 2)
            .gpu_tile(x, y, xi, yi, tile_x, tile_y);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(haar_x, haar_x)
