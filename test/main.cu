#include "NeuZephyr/ComputeGraph.cuh"
using namespace nz;


int main() {
    data::Tensor a1({9, 1}, true);
    data::Tensor real({9, 1}, true);
    float a1_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float real_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    a1.copyData(a1_data);
    real.copyData(real_data);
    nodes::io::InputNode input1(a1);
    nodes::io::InputNode input2(real);
    nodes::calc::SoftmaxNode math(&input1);
    nodes::loss::BinaryCrossEntropyNode output(&math, &input2);
    math.forward();
    output.forward();
    std::cout << output << std::endl;
    output.backward();
    math.backward();
    std::cout << input1 << std::endl;
    opt::AdaDelta optimizer(0.01);
    optimizer.step(&input1);
    std::cout << input1 << std::endl;
    return 0;
}
