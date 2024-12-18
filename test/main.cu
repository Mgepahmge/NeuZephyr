#include "NeuZephyr/ComputeGraph.cuh"
using namespace NeuZephyr;


int main() {
    Data::Tensor a1({3, 3});
    a1.fill(1.0);
    Data::Tensor b1({3, 3});
    b1.fill(2.0);
    std::cout << a1 + b1 << std::endl;
}
