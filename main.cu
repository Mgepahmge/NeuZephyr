#include <iostream>
#include "Tensor.cuh"

int main()
{
    DL::Tensor a({4, 3}, true);
    DL::Tensor b({4, 3}, true);
    DL::Tensor c({3, 4}, true);
    a.randomize(1);
    b.randomize(2);
    c.randomize(3);
    std::cout << "a:\n" << a << std::endl;
    std::cout << "b:\n" << b << std::endl;
    std::cout << "c:\n" << c << std::endl;
    std::cout << "a+b:\n" << a + b << std::endl;
    std::cout << "a-b:\n" << a - b << std::endl;
    std::cout << "a*c:\n" << a * c << std::endl;
    return 0;
}
