#include <gtest/gtest.h>
#include <random>

#include <TensorOperations.cuh>
#include <Nodes.cuh>
#include <Optimizer.cuh>
#include <ComputeGraph.cuh>
#include <Model.cuh>
using namespace nz::data;
using namespace nz::nodes;
using namespace nz::nodes::calc;
using namespace nz::nodes::io;
using namespace nz::nodes::loss;
using namespace nz;

/*
 * Note: The NeuZephyr library uses a large number of CUDA internal functions in the CUDA Kernel to accelerate operations.
 * However, the calculation precision of these internal functions is relatively poor.
 * Although the author has tried their best to avoid this situation when designing the comparison method for two Tensors,
 * some test cases still occasionally fail due to calculation precision issues.
 */

TEST(TensorBasic, TensorAdditionTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    Tensor tensor1({n, 1, h, w});
    Tensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist(gen);
    }
    tensor2.dataInject(data2.begin(), data2.end());

    Tensor result = tensor1 + tensor2;

    Tensor expected({n, c, h, w});
    std::vector<float> expected_data(n * c * h * w);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_result = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_tensor1 = ((ni * 1 + 0) * h + hi) * w + wi;
                    size_t idx_tensor2 = ((0 * c + ci) * h + hi) * w + wi;

                    expected_data[idx_result] = data1[idx_tensor1] + data2[idx_tensor2];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(TensorBasic, TensorSubtractionTest) {
    const size_t n = 3;
    const size_t c = 2;
    const size_t h = 5;
    const size_t w = 3;

    Tensor tensor1({n, 1, h, w});
    Tensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist(gen);
    }
    tensor2.dataInject(data2.begin(), data2.end());

    Tensor result = tensor1 - tensor2;

    Tensor expected({n, c, h, w});
    std::vector<float> expected_data(n * c * h * w);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_result = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_tensor1 = ((ni * 1 + 0) * h + hi) * w + wi;
                    size_t idx_tensor2 = ((0 * c + ci) * h + hi) * w + wi;

                    expected_data[idx_result] = data1[idx_tensor1] - data2[idx_tensor2];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(TensorBasic, TensorMultiplicationTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t m = 2;
    const size_t k = 3;
    const size_t p = 2;

    Tensor tensor1({n, 1, m, k});
    Tensor tensor2({1, c, k, p});

    std::vector<float> data1 = {
        // batch 0, channel 0
        1.0f, 2.0f, 3.0f, // row 0
        4.0f, 5.0f, 6.0f, // row 1

        // batch 1, channel 0
        7.0f, 8.0f, 9.0f, // row 0
        10.0f, 11.0f, 12.0f // row 1
    };
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2 = {
        // batch 0, channel 0
        1.0f, 2.0f, // row 0
        3.0f, 4.0f, // row 1
        5.0f, 6.0f, // row 2

        // batch 0, channel 1
        7.0f, 8.0f, // row 0
        9.0f, 10.0f, // row 1
        11.0f, 12.0f, // row 2

        // batch 0, channel 2
        13.0f, 14.0f, // row 0
        15.0f, 16.0f, // row 1
        17.0f, 18.0f // row 2
    };
    tensor2.dataInject(data2.begin(), data2.end());

    Tensor result = tensor1 * tensor2;

    Tensor expected({n, c, m, p});
    std::vector<float> expected_data = {
        // batch 0, channel 0
        22.0f, 28.0f,
        49.0f, 64.0f,
        // batch 0, channel 1
        58.0f, 64.0f,
        139.0f, 154.0f,
        // batch 0, channel 2
        94.0f, 100.0f,
        229.0f, 244.0f,
        // batch 1, channel 0
        76.0f, 100.0f,
        103.0f, 136.0f,
        // batch 1, channel 1
        220.0f, 244.0f,
        301.0f, 334.0f,
        // batch 1, channel 2
        364.0f, 388.0f,
        499.0f, 532.0f
    };
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(TensorBasic, TensorDivisionTest) {
    const size_t n = 2;
    const size_t c = 4;
    const size_t h = 3;
    const size_t w = 6;

    Tensor tensor1({n, 1, h, w});
    Tensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist2(1.0f, 5.0f); // 确保除数不为0

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist1(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist2(gen);
        if (dist1(gen) < 0) {
            data2[i] = -data2[i];
        }
    }
    tensor2.dataInject(data2.begin(), data2.end());

    Tensor result = tensor1 / tensor2;

    Tensor expected({n, c, h, w});
    std::vector<float> expected_data(n * c * h * w);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_result = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_tensor1 = ((ni * 1 + 0) * h + hi) * w + wi;
                    size_t idx_tensor2 = ((0 * c + ci) * h + hi) * w + wi;

                    expected_data[idx_result] = data1[idx_tensor1] / data2[idx_tensor2];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(TensorBasic, TensorTransposeTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    Tensor tensor({n, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data(n * c * h * w);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    tensor.dataInject(data.begin(), data.end());

    tensor.transpose();
    Tensor result = tensor;

    Tensor expected({n, c, w, h});
    std::vector<float> expected_data(n * c * w * h);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_tensor = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_expected = ((ni * c + ci) * w + wi) * h + hi;

                    expected_data[idx_expected] = data[idx_tensor];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(MappedTensorBasic, MappedTensorAdditionTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    MappedTensor tensor1({n, 1, h, w});
    MappedTensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist(gen);
    }
    tensor2.dataInject(data2.begin(), data2.end());

    MappedTensor result = tensor1 + tensor2;

    MappedTensor expected({n, c, h, w});
    std::vector<float> expected_data(n * c * h * w);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_result = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_tensor1 = ((ni * 1 + 0) * h + hi) * w + wi;
                    size_t idx_tensor2 = ((0 * c + ci) * h + hi) * w + wi;

                    expected_data[idx_result] = data1[idx_tensor1] + data2[idx_tensor2];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(MappedTensorBasic, MappedTensorSubtractionTest) {
    const size_t n = 3;
    const size_t c = 2;
    const size_t h = 5;
    const size_t w = 3;

    MappedTensor tensor1({n, 1, h, w});
    MappedTensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist(gen);
    }
    tensor2.dataInject(data2.begin(), data2.end());

    MappedTensor result = tensor1 - tensor2;

    MappedTensor expected({n, c, h, w});
    std::vector<float> expected_data(n * c * h * w);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_result = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_tensor1 = ((ni * 1 + 0) * h + hi) * w + wi;
                    size_t idx_tensor2 = ((0 * c + ci) * h + hi) * w + wi;

                    expected_data[idx_result] = data1[idx_tensor1] - data2[idx_tensor2];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(MappedTensorBasic, MappedTensorMultiplicationTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t m = 2;
    const size_t k = 3;
    const size_t p = 2;

    MappedTensor tensor1({n, 1, m, k});
    MappedTensor tensor2({1, c, k, p});

    std::vector<float> data1 = {
        // batch 0, channel 0
        1.0f, 2.0f, 3.0f, // row 0
        4.0f, 5.0f, 6.0f, // row 1

        // batch 1, channel 0
        7.0f, 8.0f, 9.0f, // row 0
        10.0f, 11.0f, 12.0f // row 1
    };
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2 = {
        // batch 0, channel 0
        1.0f, 2.0f, // row 0
        3.0f, 4.0f, // row 1
        5.0f, 6.0f, // row 2

        // batch 0, channel 1
        7.0f, 8.0f, // row 0
        9.0f, 10.0f, // row 1
        11.0f, 12.0f, // row 2

        // batch 0, channel 2
        13.0f, 14.0f, // row 0
        15.0f, 16.0f, // row 1
        17.0f, 18.0f // row 2
    };
    tensor2.dataInject(data2.begin(), data2.end());

    MappedTensor result = tensor1 * tensor2;

    MappedTensor expected({n, c, m, p});
    std::vector<float> expected_data = {
        // batch 0, channel 0
        22.0f, 28.0f,
        49.0f, 64.0f,
        // batch 0, channel 1
        58.0f, 64.0f,
        139.0f, 154.0f,
        // batch 0, channel 2
        94.0f, 100.0f,
        229.0f, 244.0f,
        // batch 1, channel 0
        76.0f, 100.0f,
        103.0f, 136.0f,
        // batch 1, channel 1
        220.0f, 244.0f,
        301.0f, 334.0f,
        // batch 1, channel 2
        364.0f, 388.0f,
        499.0f, 532.0f
    };
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(MappedTensorBasic, MappedTensorDivisionTest) {
    const size_t n = 2;
    const size_t c = 4;
    const size_t h = 3;
    const size_t w = 6;

    MappedTensor tensor1({n, 1, h, w});
    MappedTensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist2(1.0f, 5.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist1(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist2(gen);
        if (dist1(gen) < 0) {
            data2[i] = -data2[i];
        }
    }
    tensor2.dataInject(data2.begin(), data2.end());

    MappedTensor result = tensor1 / tensor2;

    MappedTensor expected({n, c, h, w});
    std::vector<float> expected_data(n * c * h * w);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_result = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_tensor1 = ((ni * 1 + 0) * h + hi) * w + wi;
                    size_t idx_tensor2 = ((0 * c + ci) * h + hi) * w + wi;

                    expected_data[idx_result] = data1[idx_tensor1] / data2[idx_tensor2];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(MappedTensorBasic, MappedTensorTransposeTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    MappedTensor tensor({n, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data(n * c * h * w);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    tensor.dataInject(data.begin(), data.end());

    tensor.transpose();
    MappedTensor result = tensor;

    MappedTensor expected({n, c, w, h});
    std::vector<float> expected_data(n * c * w * h);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_tensor = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_expected = ((ni * c + ci) * w + wi) * h + hi;

                    expected_data[idx_expected] = data[idx_tensor];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(TensorOperation, TransposeTestMapped) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    MappedTensor tensor({n, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data(n * c * h * w);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    tensor.dataInject(data.begin(), data.end());

    MappedTensor result = transpose(tensor);

    MappedTensor expected({n, c, w, h});
    std::vector<float> expected_data(n * c * w * h);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_tensor = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_expected = ((ni * c + ci) * w + wi) * h + hi;

                    expected_data[idx_expected] = data[idx_tensor];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(TensorOperation, TransposeTestNormal) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    Tensor tensor({n, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data(n * c * h * w);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
    tensor.dataInject(data.begin(), data.end());

    Tensor result = transpose(tensor);

    Tensor expected({n, c, w, h});
    std::vector<float> expected_data(n * c * w * h);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    size_t idx_tensor = ((ni * c + ci) * h + hi) * w + wi;
                    size_t idx_expected = ((ni * c + ci) * w + wi) * h + hi;

                    expected_data[idx_expected] = data[idx_tensor];
                }
            }
        }
    }
    expected.dataInject(expected_data.begin(), expected_data.end());

    ASSERT_TRUE(result == expected);
}

TEST(NodesBasic, AddForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    Tensor tensor1({n, 1, h, w});
    Tensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist(gen);
    }
    tensor2.dataInject(data2.begin(), data2.end());

    Tensor result = tensor1 + tensor2;
    InputNode input1({n, 1, h, w});
    InputNode input2({1, c, h, w});
    input1.dataInject(data1.begin(), data1.end());
    input2.dataInject(data2.begin(), data2.end());
    AddNode add(&input1, &input2);
    add.forward();
    EXPECT_EQ(*add.output, result);
}

TEST(NodesBasic, AddBackwardSpecial) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    InputNode input1({n, 1, h, w}, true);
    InputNode input2({1, c, h, w}, true);
    AddNode add(&input1, &input2);
    std::vector<float> addGrad(n * c * h * w, 1);
    std::vector<float> grad1(n * 1 * h * w, c);
    std::vector<float> grad2(1 * c * h * w, n);
    add.output->dataInject(addGrad.begin(), addGrad.end(), true);
    add.backward();
    Tensor expectedGrad1({n, 1, h, w}, true);
    Tensor expectedGrad2({1, c, h, w}, true);
    expectedGrad1.dataInject(grad1.begin(), grad1.end(), true);
    expectedGrad2.dataInject(grad2.begin(), grad2.end(), true);
    EXPECT_EQ(*input1.output, expectedGrad1);
    EXPECT_EQ(*input2.output, expectedGrad2);
}

TEST(NodesBasic, AddBackwardNormal) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    InputNode inputData({n, c, h, w}, true);
    InputNode Weights({1, c, h, w}, true);
    AddNode add(&inputData, &Weights);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> addGrad(n * c * h * w);
    std::vector<float> weightsGrad(1 * c * h * w);
    for (float & i : addGrad) {
        i = dist(gen);
    }
    for (auto i = 0; i < c; i++) {
        for (auto j = 0; j < h; j++) {
            for (auto k = 0; k < w; k++) {
                for (auto l =0; l <n; l++) {
                    weightsGrad[i * (h * w) + j * w + k] += addGrad[l * (c * h * w) +i * (h * w) + j * w + k];
                }
            }
        }
    }
    add.output->dataInject(addGrad.begin(), addGrad.end(), true);
    add.backward();
    Tensor expectedGradInput({n, c, h, w}, true);
    expectedGradInput.dataInject(addGrad.begin(), addGrad.end(), true);
    Tensor expectedGradWeights({1, c, h, w}, true);
    expectedGradWeights.dataInject(weightsGrad.begin(), weightsGrad.end(), true);
    EXPECT_EQ(*inputData.output, expectedGradInput);
    EXPECT_EQ(*Weights.output, expectedGradWeights);
}

TEST(TensorCore, TensorCoreGEMMTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t m = 2;
    const size_t k = 3;
    const size_t p = 2;

    Tensor tensor1({n, 1, m, k});
    Tensor tensor2({1, c, k, p});

    std::vector<float> data1 = {
        // batch 0, channel 0
        1.0f, 2.0f, 3.0f, // row 0
        4.0f, 5.0f, 6.0f, // row 1

        // batch 1, channel 0
        7.0f, 8.0f, 9.0f, // row 0
        10.0f, 11.0f, 12.0f // row 1
    };
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2 = {
        // batch 0, channel 0
        1.0f, 2.0f, // row 0
        3.0f, 4.0f, // row 1
        5.0f, 6.0f, // row 2

        // batch 0, channel 1
        7.0f, 8.0f, // row 0
        9.0f, 10.0f, // row 1
        11.0f, 12.0f, // row 2

        // batch 0, channel 2
        13.0f, 14.0f, // row 0
        15.0f, 16.0f, // row 1
        17.0f, 18.0f // row 2
    };
    tensor2.dataInject(data2.begin(), data2.end());

    auto result1 = tensor1 * tensor2;
    Tensor result2({n,c, m, p});
    GEMMTensorCore(result2, tensor1, tensor2);
    EXPECT_EQ(result1, result2);
}

TEST(NodeBasic, MatMulForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t m = 2;
    const size_t k = 3;
    const size_t p = 2;

    InputNode input1({n ,1, m, k});
    InputNode input2({1, c, k, p});

    std::vector<float> data1 = {
        // batch 0, channel 0
        1.0f, 2.0f, 3.0f, // row 0
        4.0f, 5.0f, 6.0f, // row 1

        // batch 1, channel 0
        7.0f, 8.0f, 9.0f, // row 0
        10.0f, 11.0f, 12.0f // row 1
    };
    input1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2 = {
        // batch 0, channel 0
        1.0f, 2.0f, // row 0
        3.0f, 4.0f, // row 1
        5.0f, 6.0f, // row 2

        // batch 0, channel 1
        7.0f, 8.0f, // row 0
        9.0f, 10.0f, // row 1
        11.0f, 12.0f, // row 2

        // batch 0, channel 2
        13.0f, 14.0f, // row 0
        15.0f, 16.0f, // row 1
        17.0f, 18.0f // row 2
    };
    input2.dataInject(data2.begin(), data2.end());

    MatMulNode mul(&input1, &input2);
    mul.forward();

    std::vector expected_data = {
        // batch 0, channel 0
        22.0f, 28.0f,
        49.0f, 64.0f,
        // batch 0, channel 1
        58.0f, 64.0f,
        139.0f, 154.0f,
        // batch 0, channel 2
        94.0f, 100.0f,
        229.0f, 244.0f,
        // batch 1, channel 0
        76.0f, 100.0f,
        103.0f, 136.0f,
        // batch 1, channel 1
        220.0f, 244.0f,
        301.0f, 334.0f,
        // batch 1, channel 2
        364.0f, 388.0f,
        499.0f, 532.0f
    };
    Tensor expected({n, c, m, p});
    expected.dataInject(expected_data.begin(), expected_data.end());
    EXPECT_EQ(*mul.output, expected);
}

TEST(NodeBasic, MatMulBackwardSpecial) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t m = 2;
    const size_t k = 3;
    const size_t p = 2;

    InputNode input1({n ,1, m, k}, true);
    InputNode input2({1, c, k, p}, true);
    input1.output->fill(1);
    input2.output->fill(1);
    MatMulNode mul(&input1, &input2);
    mul.output->fill(1, true);
    mul.backward();
    Tensor expected1({n, 1, m , k}, true);
    Tensor expected2({1, c, k, p}, true);
    expected1.fill(1);
    expected1.fill(2 * c, true);
    expected2.fill(1);
    expected2.fill(2 * n, true);
    EXPECT_EQ(expected1, *input1.output);
    EXPECT_EQ(expected2, *input2.output);
}

TEST(NodeBasic, MatMulBackwardNormal) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t m = 2;
    const size_t k = 3;
    const size_t p = 2;

    std::vector<float> Adata = {// batch0
        /* channel0 */ 1,2,3,4,5,6,      // 2x3 = [[1,2,3],[4,5,6]]
        /* channel1 */ 2,3,4,5,6,7,      // 2x3 = [[2,3,4],[5,6,7]]
        /* channel2 */ 3,4,5,6,7,8,      // 2x3 = [[3,4,5],[6,7,8]]
        // batch1（与batch0相同）
        /* channel0 */ 1,2,3,4,5,6,
        /* channel1 */ 2,3,4,5,6,7,
        /* channel2 */ 3,4,5,6,7,8};

    std::vector<float> Bdata = {
        /* channel0 */ 1,4,2,5,3,6,
        /* channel1 */ 2,5,3,6,4,7,      // 3x2 = [[2,5],[3,6],[4,7]]
        /* channel2 */ 3,6,4,7,5,8};      // 3x2 = [[3,6],[4,7],[5,8]]

    std::vector<float> Cgrad = {// batch0
        1,1,1,1,  // channel0 (2x2)
        1,1,1,1,  // channel1 (2x2)
        1,1,1,1,  // channel2 (2x2)
        // batch1 (same as batch0)
        1,1,1,1,
        1,1,1,1,
        1,1,1,1};

    std::vector<float> Agrad = {// batch0
        /* channel0 */ 5, 7, 9, 5, 7 ,9,
        /* channel1 */ 7, 9, 11, 7, 9, 11,
        /* channel2 */ 9, 11, 13, 9, 11, 13,
        // batch1
        /* channel0 */ 5, 7, 9, 5, 7 ,9,
        /* channel1 */ 7, 9, 11, 7, 9, 11,
        /* channel2 */ 9, 11, 13, 9, 11, 13};

    std::vector<float> Bgrad = {
        /* channel0 */ 10,10,14,14,18,18,
        /* channel1 */ 14,14,18,18,22,22,
        /* channel2 */ 18,18,22,22,26,26};

    InputNode input1({n ,c, m, k}, true);
    InputNode input2({1, c, k, p}, true);
    input1.dataInject(Adata.begin(), Adata.end());
    input2.dataInject(Bdata.begin(), Bdata.end());
    MatMulNode mul(&input1, &input2);
    mul.output->dataInject(Cgrad.begin(), Cgrad.end(), true);
    mul.backward();
    Tensor expected1({n, c, m , k}, true);
    Tensor expected2({1, c, k, p}, true);
    expected1.dataInject(Adata.begin(), Adata.end());
    expected1.dataInject(Agrad.begin(), Agrad.end(), true);
    expected2.dataInject(Bdata.begin(), Bdata.end());
    expected2.dataInject(Bgrad.begin(), Bgrad.end(), true);
    EXPECT_EQ(expected1, *input1.output);
    EXPECT_EQ(expected2, *input2.output);
}

TEST(NodesBasic, SubForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    Tensor tensor1({n, 1, h, w});
    Tensor tensor2({1, c, h, w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> data1(n * 1 * h * w);
    for (size_t i = 0; i < data1.size(); ++i) {
        data1[i] = dist(gen);
    }
    tensor1.dataInject(data1.begin(), data1.end());

    std::vector<float> data2(1 * c * h * w);
    for (size_t i = 0; i < data2.size(); ++i) {
        data2[i] = dist(gen);
    }
    tensor2.dataInject(data2.begin(), data2.end());

    Tensor result = tensor1 - tensor2;
    InputNode input1({n, 1, h, w});
    InputNode input2({1, c, h, w});
    input1.dataInject(data1.begin(), data1.end());
    input2.dataInject(data2.begin(), data2.end());
    SubNode sub(&input1, &input2);
    sub.forward();
    EXPECT_EQ(*sub.output, result);
}

TEST(NodesBasic, SubBackwardSpecial) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    InputNode input1({n, 1, h, w}, true);
    InputNode input2({1, c, h, w}, true);
    SubNode sub(&input1, &input2);
    std::vector<float> subGrad(n * c * h * w, 1);
    std::vector<float> grad1(n * 1 * h * w, 3);
    std::vector<float> grad2(1 * c * h * w, -2);
    sub.output->dataInject(subGrad.begin(), subGrad.end(), true);
    sub.backward();
    Tensor expectedGrad1({n, 1, h, w}, true);
    Tensor expectedGrad2({1, c, h, w}, true);
    expectedGrad1.dataInject(grad1.begin(), grad1.end(), true);
    expectedGrad2.dataInject(grad2.begin(), grad2.end(), true);
    EXPECT_EQ(*input1.output, expectedGrad1);
    EXPECT_EQ(*input2.output, expectedGrad2);
}

TEST(NodesBasic, SubBackwardNormal) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    InputNode inputData({n, c, h, w}, true);
    InputNode Weights({1, c, h, w}, true);
    SubNode sub(&inputData, &Weights);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> subGrad(n * c * h * w);
    std::vector<float> weightsGrad(1 * c * h * w);
    for (float & i : subGrad) {
        i = dist(gen);
    }
    for (auto i = 0; i < c; i++) {
        for (auto j = 0; j < h; j++) {
            for (auto k = 0; k < w; k++) {
                for (auto l =0; l <n; l++) {
                    weightsGrad[i * (h * w) + j * w + k] -= subGrad[l * (c * h * w) +i * (h * w) + j * w + k];
                }
            }
        }
    }
    sub.output->dataInject(subGrad.begin(), subGrad.end(), true);
    sub.backward();
    Tensor expectedGradInput({n, c, h, w}, true);
    expectedGradInput.dataInject(subGrad.begin(), subGrad.end(), true);
    Tensor expectedGradWeights({1, c, h, w}, true);
    expectedGradWeights.dataInject(weightsGrad.begin(), weightsGrad.end(), true);
    EXPECT_EQ(*inputData.output, expectedGradInput);
    EXPECT_EQ(*Weights.output, expectedGradWeights);
}

TEST(TensorBasic, TensorSoftmaxTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            float sum = 0.0f;
            for (auto k = 0; k < h; k++) {
                sum += std::exp(inputData[i * (c * h * w) + j * (h * w) + k]);
            }
            for (auto k = 0; k < h; k++) {
                expectedData[i * (c * h * w) + j * (h * w) + k] = std::exp(inputData[i * (c * h * w) + j * (h * w) + k]) / sum;
            }
        }
    }
    input.dataInject(inputData.begin(), inputData.end());
    auto result = Softmax(input);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(TensorBasic, TensorReLUTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] > 0 ? inputData[i] : 0;
    }
    input.dataInject(inputData.begin(), inputData.end());
    auto result = ReLU(input);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, ReLUForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] > 0 ? inputData[i] : 0;
    }
    input.dataInject(inputData.begin(), inputData.end());
    ReLUNode result(&input);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, ReLUBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        expectedGrad[i] = inputData[i] > 0 ? grad[i] : 0;
    }
    input.dataInject(inputData.begin(), inputData.end());
    ReLUNode result(&input);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorSigmoidTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = 1 / (1 + std::exp(-inputData[i]));
    }
    input.dataInject(inputData.begin(), inputData.end());
    auto result = Sigmoid(input);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, SigmoidForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = 1 / (1 + std::exp(-inputData[i]));
    }
    input.dataInject(inputData.begin(), inputData.end());
    SigmoidNode result(&input);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, SigmoidBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = 1 / (1 + std::exp(-inputData[i]));
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        expectedGrad[i] = grad[i] * outputData[i] * (1 - outputData[i]);
    }
    input.dataInject(inputData.begin(), inputData.end());
    SigmoidNode result(&input);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorTanhTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = (std::exp(inputData[i]) - std::exp(-inputData[i])) / (std::exp(inputData[i]) + std::exp(-inputData[i]));
    }

    input.dataInject(inputData.begin(), inputData.end());
    auto result = Tanh(input);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, TanhForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = (std::exp(inputData[i]) - std::exp(-inputData[i])) / (std::exp(inputData[i]) + std::exp(-inputData[i]));
    }
    input.dataInject(inputData.begin(), inputData.end());
    TanhNode result(&input);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, TanhBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 5;
    const size_t w = 5;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = (std::exp(inputData[i]) - std::exp(-inputData[i])) / (std::exp(inputData[i]) + std::exp(-inputData[i]));
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        expectedGrad[i] = grad[i] * (1 - outputData[i] * outputData[i]);
    }
    input.dataInject(inputData.begin(), inputData.end());
    TanhNode result(&input);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorLeakyReLUTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.01f;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] > 0 ? inputData[i] : alpha * inputData[i];
    }

    input.dataInject(inputData.begin(), inputData.end());
    auto result = LeakyReLU(input, alpha);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, LeakyReLUForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.01f;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] > 0 ? inputData[i] : alpha * inputData[i];
    }
    input.dataInject(inputData.begin(), inputData.end());
    LeakyReLUNode result(&input, alpha);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, LeakyReLUBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.01f;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = inputData[i] > 0 ? inputData[i] : alpha * inputData[i];
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        expectedGrad[i] = grad[i] * (inputData[i] > 0 ? 1 : alpha);
    }
    input.dataInject(inputData.begin(), inputData.end());
    LeakyReLUNode result(&input, alpha);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorSwishTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] / (1 + std::exp(-inputData[i]));
    }

    input.dataInject(inputData.begin(), inputData.end());
    auto result = Swish(input);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, SwishForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] / (1 + std::exp(-inputData[i]));
    }
    input.dataInject(inputData.begin(), inputData.end());
    SwishNode result(&input);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, SwishBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = inputData[i] / (1 + std::exp(-inputData[i]));
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        expectedGrad[i] = outputData[i] * (1 - outputData[i]) * grad[i] + 1.0f / (1.0f + std::exp(-inputData[i]));
    }
    input.dataInject(inputData.begin(), inputData.end());
    SwishNode result(&input);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorELUTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 1.0f;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] > 0 ? inputData[i] : alpha * (std::exp(inputData[i]) - 1);
    }

    input.dataInject(inputData.begin(), inputData.end());
    auto result = ELU(input, alpha);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, ELUForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 1.0f;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] > 0 ? inputData[i] : alpha * (std::exp(inputData[i]) - 1);
    }
    input.dataInject(inputData.begin(), inputData.end());
    ELUNode result(&input, alpha);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, ELUBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 1.0f;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = inputData[i] > 0 ? inputData[i] : alpha * (std::exp(inputData[i]) - 1);
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        expectedGrad[i] = outputData[i] > 0 ? grad[i] : alpha * grad[i] * std::exp(inputData[i]);
    }
    input.dataInject(inputData.begin(), inputData.end());
    ELUNode result(&input, alpha);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorHardSigmoidTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.2f;
    const float beta = 0.5f;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = std::min(std::max(alpha * inputData[i] + beta, 0.0f), 1.0f);
    }

    input.dataInject(inputData.begin(), inputData.end());
    auto result = HardSigmoid(input, alpha, beta);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, HardSigmoidForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.2f;
    const float beta = 0.5f;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = std::min(std::max(alpha * inputData[i] + beta, 0.0f), 1.0f);
    }
    input.dataInject(inputData.begin(), inputData.end());
    HardSigmoidNode result(&input, alpha, beta);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, HardSigmoidBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.2f;
    const float beta = 0.5f;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = std::min(std::max(alpha * inputData[i] + beta, 0.0f), 1.0f);
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        const float x = inputData[i] * alpha + beta;
        if (x > 0.0f && x < 1.0f) {
            expectedGrad[i] = grad[i] * alpha;
        } else {
            expectedGrad[i] = 0.0f;
        }
    }
    input.dataInject(inputData.begin(), inputData.end());
    HardSigmoidNode result(&input, alpha, beta);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(TensorBasic, TensorHardSwishTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.5f;
    const float beta = 0.5f;
    Tensor input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] * std::min(std::max(inputData[i] * alpha + beta, 0.0f), 1.0f);
    }

    input.dataInject(inputData.begin(), inputData.end());
    auto result = HardSwish(input, alpha, beta);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, HardSwishForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.5f;
    const float beta = 0.5f;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] * std::min(std::max(inputData[i] * alpha + beta, 0.0f), 1.0f);
    }
    input.dataInject(inputData.begin(), inputData.end());
    HardSwishNode result(&input, alpha, beta);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, HardSwishBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 16;
    const float alpha = 0.5f;
    const float beta = 0.5f;
    InputNode input({n, c, h, w}, true);
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> outputData(n*c*h*w);
    std::vector<float> grad(n*c*h*w);
    std::vector<float> expectedGrad(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0 ; i < inputData.size(); i++) {
        outputData[i] = inputData[i] * std::min(std::max(inputData[i] * alpha + beta, 0.0f), 1.0f);
    }
    for (auto i = 0 ; i < grad.size(); i++) {
        const float x = std::min(std::max(inputData[i] * alpha + beta, 0.0f), 1.0f);
        expectedGrad[i] = x + grad[i] * inputData[i] * alpha * (1.0f - x);
    }
    input.dataInject(inputData.begin(), inputData.end());
    HardSwishNode result(&input, alpha, beta);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.forward();
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputData.begin(), inputData.end());
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(NodeBasic, SoftmaxForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;
    InputNode input({n, c, h, w});
    std::vector<float> inputData(n*c*h*w);
    std::vector<float> expectedData(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            float sum = 0.0f;
            for (auto k = 0; k < h; k++) {
                sum += std::exp(inputData[i * (c * h * w) + j * (h * w) + k]);
            }
            for (auto k = 0; k < h; k++) {
                expectedData[i * (c * h * w) + j * (h * w) + k] = std::exp(inputData[i * (c * h * w) + j * (h * w) + k]) / sum;
            }
        }
    }
    input.dataInject(inputData.begin(), inputData.end());
    SoftmaxNode softmax(&input);
    softmax.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *softmax.output);
}

TEST(TensorBasic, SoftmaxJacobianTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> outputData(n * c * h * h);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            const size_t offset_i = i * (c * h * w) + j * (h * w);
            const size_t offset_o = i * (c * h * h) + j * (h * h);
            for (auto x = 0; x < h; x++) {
                for (auto y = 0; y < h; y++) {
                    if (x == y) {
                        outputData[offset_o + x * h + y] = inputData[x + offset_i] * (1 - inputData[x + offset_i]);
                    } else {
                        outputData[offset_o + x * h + y] = -inputData[x + offset_i] * inputData[y + offset_i];
                    }
                }
            }
        }
    }
    Tensor input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    auto output = softmaxJacobian(input);
    Tensor expected({n, c, h, h});
    expected.dataInject(outputData.begin(), outputData.end());
    EXPECT_EQ(output, expected);
}

TEST(NodeBasic, SoftmaxBackwardRow) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> outputData(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> jacobianData(n * c * h * h);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            float sum = 0.0f;
            for (auto k = 0; k < h; k++) {
                sum += std::exp(inputData[i * (c * h * w) + j * (h * w) + k]);
            }
            for (auto k = 0; k < h; k++) {
                outputData[i * (c * h * w) + j * (h * w) + k] = std::exp(inputData[i * (c * h * w) + j * (h * w) + k]) / sum;
            }
        }
    }

    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            const size_t offset_i = i * (c * h * w) + j * (h * w);
            const size_t offset_o = i * (c * h * h) + j * (h * h);
            for (auto x = 0; x < h; x++) {
                for (auto y = 0; y < h; y++) {
                    if (x == y) {
                        jacobianData[offset_o + x * h + y] = outputData[x + offset_i] * (1 - outputData[x + offset_i]);
                    } else {
                        jacobianData[offset_o + x * h + y] = -outputData[x + offset_i] * outputData[y + offset_i];
                    }
                }
            }
        }
    }
    MappedTensor jacobian({n, c, h, h});
    MappedTensor outputGrad({n, c, h, w});
    jacobian.dataInject(jacobianData.begin(), jacobianData.end());
    outputGrad.dataInject(grad.begin(), grad.end());
    MappedTensor inputGrad ({n, c, h, w});
    GEMMTensorCore(inputGrad, jacobian, outputGrad);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(inputGrad.begin(), inputGrad.end(), true);
    expected.dataInject(inputData.begin(), inputData.end());
    InputNode input({n, c, h, w}, true);
    input.dataInject(inputData.begin(), inputData.end());
    SoftmaxNode softmax(&input);
    softmax.forward();
    softmax.output->dataInject(grad.begin(), grad.end(), true);
    softmax.backward();
    EXPECT_EQ(*input.output, expected);
}

TEST(NodeLoss, MSEForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 5;

    std::vector<float> predict(n * c * h * w);
    std::vector<float> target(n * c * h * w);
    float loss = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : predict) {
        i = dist(gen);
    }
    for (auto& i : target) {
        i = dist(gen);
    }
    for (auto i = 0; i < predict.size(); i++) {
        loss += (predict[i] - target[i]) * (predict[i] - target[i]);
    }
    loss /= (n * c * h * w);
    InputNode input1({n, c, h, w});
    InputNode input2({n, c, h, w});
    input1.dataInject(predict.begin(), predict.end());
    input2.dataInject(target.begin(), target.end());
    MeanSquaredErrorNode mse(&input1, &input2);
    mse.forward();
    EXPECT_NEAR(mse.getLoss(), loss, 1e-2);
}

TEST(NodeLoss, MSEBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 5;

    std::vector<float> predict(n * c * h * w);
    std::vector<float> target(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : predict) {
        i = dist(gen);
    }
    for (auto& i : target) {
        i = dist(gen);
    }
    for (auto i = 0; i < predict.size(); i++) {
        grad[i] = 2 * (predict[i] - target[i]) / (n * c * h * w);
    }
    InputNode input1({n, c, h, w}, true);
    InputNode input2({n, c, h, w});
    input1.dataInject(predict.begin(), predict.end());
    input2.dataInject(target.begin(), target.end());
    MeanSquaredErrorNode mse(&input1, &input2);
    mse.forward();
    mse.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(predict.begin(), predict.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*mse.output, expected);
}

TEST(NodeLoss, BCEForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 5;

    std::vector<float> predict(n * c * h * w);
    std::vector<float> target(n * c * h * w);
    float loss = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    for (auto& i : predict) {
        i = dist(gen);
    }
    for (auto& i : target) {
        i = dist(gen);
    }
    for (auto i = 0; i < predict.size(); i++) {
        loss += -target[i] * std::log(predict[i]) - (1 - target[i]) * std::log(1 - predict[i]);
    }
    loss /= (n * c * h * w);
    InputNode input1({n, c, h, w});
    InputNode input2({n, c, h, w});
    input1.dataInject(predict.begin(), predict.end());
    input2.dataInject(target.begin(), target.end());
    BinaryCrossEntropyNode mse(&input1, &input2);
    mse.forward();
    EXPECT_NEAR(mse.getLoss(), loss, 1e-2);
}

TEST(NodeLoss, BCEBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 5;

    std::vector<float> predict(n * c * h * w);
    std::vector<float> target(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    for (auto& i : predict) {
        i = dist(gen);
    }
    for (auto& i : target) {
        i = dist(gen);
    }
    for (auto i = 0; i < predict.size(); i++) {
        grad[i] = ((predict[i] - target[i]) / (predict[i] * (1 - predict[i]))) / (n * c * h * w);
    }
    InputNode input1({n, c, h, w}, true);
    InputNode input2({n, c, h, w});
    input1.dataInject(predict.begin(), predict.end());
    input2.dataInject(target.begin(), target.end());
    BinaryCrossEntropyNode mse(&input1, &input2);
    mse.forward();
    mse.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(predict.begin(), predict.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*mse.output, expected);
}

TEST(NodeBasic, ScalarAddForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    const float scalar = 2.0f;
    for (auto i = 0; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] + scalar;
    }
    InputNode input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    ScalarAddNode result(&input, scalar);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, ScalarAddBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;

    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedGrad(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < grad.size(); i++) {
        expectedGrad[i] = grad[i];
    }

    InputNode input({n, c, h, w}, true);
    const float scalar = 2.0f;
    ScalarAddNode result(&input, scalar);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(NodeBasic, ScalarSubForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    const float scalar = 2.0f;
    for (auto i = 0; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] - scalar;
    }
    InputNode input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    ScalarSubNode result(&input, scalar);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, ScalarSubBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;

    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedGrad(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < grad.size(); i++) {
        expectedGrad[i] = grad[i];
    }

    InputNode input({n, c, h, w}, true);
    const float scalar = 2.0f;
    ScalarSubNode result(&input, scalar);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(NodeBasic, ScalarMulForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    const float scalar = 2.0f;
    for (auto i = 0; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] * scalar;
    }
    InputNode input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    ScalarMulNode result(&input, scalar);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, ScalarMulBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float scalar = 2.0f;

    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedGrad(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < grad.size(); i++) {
        expectedGrad[i] = grad[i] * scalar;
    }

    InputNode input({n, c, h, w}, true);
    ScalarMulNode result(&input, scalar);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(NodeBasic, ScalarDivForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    const float scalar = 2.0f;
    for (auto i = 0; i < inputData.size(); i++) {
        expectedData[i] = inputData[i] / scalar;
    }
    InputNode input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    ScalarDivNode result(&input, scalar);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, ScalarDivBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float scalar = 2.0f;

    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedGrad(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < grad.size(); i++) {
        expectedGrad[i] = grad[i] / scalar;
    }

    InputNode input({n, c, h, w}, true);
    ScalarDivNode result(&input, scalar);
    result.output->dataInject(grad.begin(), grad.end(), true);
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, SGDTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.01f;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < data.size(); i++) {
        expectedData[i] = data[i] - learningRate * grad[i];
    }

    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::SGD optimizer(learningRate);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, MomentumTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.01f;
    const float beta = 0.9f;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::vector<float> velocity(n * c * h * w, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < data.size(); i++) {
        velocity[i] = beta * velocity[i] + (1 - beta) * grad[i];
        expectedData[i] = data[i] - learningRate * velocity[i];
    }
    for (auto i = 0; i < data.size(); i++) {
        velocity[i] = beta * velocity[i] + (1 - beta) * grad[i];
        expectedData[i] = expectedData[i] - learningRate * velocity[i];
    }

    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::Momentum optimizer(learningRate, beta);
    optimizer.step(&input);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, AdaGrad) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.01f;
    const float epsilon = 1e-6f;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::vector<float> G(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < data.size(); i++) {
        G[i] = G[i] + grad[i] * grad[i];
        expectedData[i] = data[i] - (learningRate / (std::sqrt(G[i]) + epsilon)) * grad[i];
    }
    for (auto i = 0; i < data.size(); i++) {
        G[i] = G[i] + grad[i] * grad[i];
        expectedData[i] = expectedData[i] - (learningRate / (std::sqrt(G[i]) + epsilon)) * grad[i];
    }

    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::AdaGrad optimizer(learningRate);
    optimizer.step(&input);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, RMSPropTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.01f;
    const float decay_rate = 0.9f;
    const float epsilon = 1e-6f;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::vector<float> v(n * c * h * w, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < data.size(); i++) {
        v[i] = decay_rate * v[i] + (1 - decay_rate) * grad[i] * grad[i];
        expectedData[i] = data[i] - (learningRate / (std::sqrt(v[i]) + epsilon)) * grad[i];
    }
    for (auto i = 0; i < data.size(); i++) {
        v[i] = decay_rate * v[i] + (1 - decay_rate) * grad[i] * grad[i];
        expectedData[i] = expectedData[i] - (learningRate / (std::sqrt(v[i]) + epsilon)) * grad[i];
    }
    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::RMSprop optimizer(learningRate, decay_rate);
    optimizer.step(&input);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, AdamTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-6f;
    int it = 0;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::vector<float> m(n * c * h * w, 0.0f);
    std::vector<float> v(n * c * h * w, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    it++;
    for (auto i = 0; i < data.size(); i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float modified_m = m[i] / (1 - std::pow(beta1, it));
        float modified_v = v[i] / (1 - std::pow(beta2, it));
        expectedData[i] = data[i] - (learningRate / (std::sqrt(modified_v) + epsilon)) * modified_m;
    }
    it++;
    for (auto i = 0; i < data.size(); i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float modified_m = m[i] / (1 - std::pow(beta1, it));
        float modified_v = v[i] / (1 - std::pow(beta2, it));
        expectedData[i] = expectedData[i] - (learningRate / (std::sqrt(modified_v) + epsilon)) * modified_m;
    }
    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::Adam optimizer(learningRate, beta1, beta2);
    optimizer.step(&input);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, NAdamTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-6f;
    int it = 0;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::vector<float> m(n * c * h * w, 0.0f);
    std::vector<float> m_mod(n * c * h * w, 0.0f);
    std::vector<float> v(n * c * h * w, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    it++;
    for (auto i = 0; i < data.size(); i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float modified_m = m[i] / (1 - std::pow(beta1, it));
        float modified_v = v[i] / (1 - std::pow(beta2, it));
        float modified_m_minus_1 = m_mod[i] * beta1 + (1 - beta1) * grad[i];
        expectedData[i] = data[i] - (learningRate / (std::sqrt(modified_v) + epsilon)) * modified_m_minus_1;
        m_mod[i] = modified_m;
    }
    it++;
    for (auto i = 0; i < data.size(); i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float modified_m = m[i] / (1 - std::pow(beta1, it));
        float modified_v = v[i] / (1 - std::pow(beta2, it));
        float modified_m_minus_1 = m_mod[i] * beta1 + (1 - beta1) * grad[i];
        expectedData[i] = expectedData[i] - (learningRate / (std::sqrt(modified_v) + epsilon)) * modified_m_minus_1;
        m_mod[i] = modified_m;
    }
    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::NAdam optimizer(learningRate, beta1, beta2);
    optimizer.step(&input);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(OptimizerBasic, AdaDeltaTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 3;
    const size_t w = 4;
    const float learningRate = 0.9f;
    const float epsilon = 1e-6f;

    std::vector<float> data(n * c * h * w);
    std::vector<float> grad(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);
    std::vector<float> acc(n * c * h * w, 0.0f);
    std::vector<float> acc_grad(n * c * h * w, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : data) {
        i = dist(gen);
    }
    for (auto& i : grad) {
        i = dist(gen);
    }
    for (auto i = 0; i < data.size(); i++) {
        acc_grad[i] = learningRate * acc_grad[i] + (1 - learningRate) * grad[i] * grad[i];
        float theta = -grad[i] * std::sqrt(acc[i] + epsilon) / std::sqrt(acc_grad[i] + epsilon);
        expectedData[i] = data[i] + theta;
        acc[i] = learningRate * acc[i] + (1 - learningRate) * theta * theta;
    }
    for (auto i = 0; i < data.size(); i++) {
        acc_grad[i] = learningRate * acc_grad[i] + (1 - learningRate) * grad[i] * grad[i];
        float theta = -grad[i] * std::sqrt(acc[i] + epsilon) / std::sqrt(acc_grad[i] + epsilon);
        expectedData[i] = expectedData[i] + theta;
        acc[i] = learningRate * acc[i] + (1 - learningRate) * theta * theta;
    }
    InputNode input({n, c, h, w}, true);
    input.dataInject(data.begin(), data.end());
    input.dataInject(grad.begin(), grad.end(), true);
    opt::AdaDelta optimizer(learningRate);
    optimizer.step(&input);
    optimizer.step(&input);
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(grad.begin(), grad.end(), true);
    EXPECT_EQ(*input.output, expected);
}

TEST(ComputeGraph, GraphForwardTest) {
    graph::ComputeGraph graph;
    InputNode input1({2, 3, 4, 5});
    InputNode param1({1, 3, 5, 1});
    InputNode param2({2, 3, 4, 1});
    InputNode target({2, 3, 4, 1});
    MatMulNode matmul(&input1, &param1);
    ReLUNode relu(&matmul);
    AddNode add(&relu, &param2);
    MeanSquaredErrorNode mse(&add, &target);
    MappedTensor input1Data({2, 3, 4, 5});
    MappedTensor param1Data({1, 3, 5, 1});
    MappedTensor param2Data({2, 3, 4, 1});
    MappedTensor targetData({2, 3, 4, 1});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : input1Data) {
        i = dist(gen);
    }
    for (auto& i : param1Data) {
        i = dist(gen);
    }
    for (auto& i : param2Data) {
        i = dist(gen);
    }
    for (auto& i : targetData) {
        i = dist(gen);
    }
    input1.dataInject(input1Data.begin(), input1Data.end());
    param1.dataInject(param1Data.begin(), param1Data.end());
    param2.dataInject(param2Data.begin(), param2Data.end());
    target.dataInject(targetData.begin(), targetData.end());
    MappedTensor mulResult({2, 3, 4, 1});
    GEMMTensorCore(mulResult, input1Data, param1Data);
    auto reluResult = ReLU(mulResult);
    auto addResult = reluResult + param2Data;
    float loss = 0.0f;
    for (auto i = 0; i < addResult.size(); i++) {
        loss += (addResult[i] - targetData[i]) * (addResult[i] - targetData[i]);
    }
    loss /= static_cast<float>(addResult.size());
    graph.addNode(&input1);
    graph.addNode(&param1);
    graph.addNode(&param2);
    graph.addNode(&target);
    graph.addNode(&matmul);
    graph.addNode(&relu);
    graph.addNode(&add);
    graph.addNode(&mse);
    graph.forward();
    Tensor expected({2, 3, 4, 1});
    expected.dataInject(addResult.begin(), addResult.end());
    EXPECT_EQ(expected, *add.output);
    EXPECT_NEAR(loss, mse.getLoss(), 1e-2);
}

TEST(NodeBasic, ReshapeForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> data(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : data) {
        i = dist(gen);
    }
    Tensor tensor({n, c, h, w});
    InputNode input({n, c, h, w});
    tensor.dataInject(data.begin(), data.end());
    input.dataInject(data.begin(), data.end());
    ReshapeNode result(&input, {n, 1, c*h*w, 1});
    result.forward();
    tensor.reshape({n, 1, c*h*w, 1});
    EXPECT_EQ(tensor, *result.output);
}

TEST(NodeBasic, ReshapeBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> data(n*c*h*w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : data) {
        i = dist(gen);
    }
    Tensor tensor({n, c, h, w}, true);
    InputNode input({n, c, h, w}, true);
    tensor.dataInject(data.begin(), data.end(), true);
    ReshapeNode result(&input, {n, 1, c*h*w, 1});
    result.dataInject(data.begin(), data.end(), true);
    result.backward();
    EXPECT_EQ(tensor, *input.output);
}

TEST(NodeBasic, ExpandForward) {
    const size_t n = 10;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> input(1 * c * h * w);
    std::vector<float> expected(n * c * h * w);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& i : input) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c * h * w; j++) {
            expected[i * c * h * w + j] = input[j];
        }
    }
    InputNode i({1, c, h, w});
    i.dataInject(input.begin(), input.end());
    ExpandNode result(&i, n);
    result.forward();
    Tensor expectedTensor({n, c, h, w});
    expectedTensor.dataInject(expected.begin(), expected.end());
    EXPECT_EQ(expectedTensor, *result.output);
}

TEST(NodeBasic, ExpandBackward) {
    const size_t n = 10;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> input(n * c * h * w);
    std::vector<float> expected(1 * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& i : input) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c * h * w; j++) {
            expected[j] += input[i * c * h * w + j];
        }
    }
    InputNode i({1, c, h, w}, true);
    ExpandNode result(&i, n);
    result.dataInject(input.begin(), input.end(), true);
    result.backward();
    Tensor expectedTensor({1, c, h, w}, true);
    expectedTensor.dataInject(expected.begin(), expected.end(), true);
    EXPECT_EQ(expectedTensor, *i.output);
}

class MSETestModel : public Model {
public:
    InputNode input;
    InputNode target;

    MSETestModel() : input({5, 3, 12, 1}, true), target({1, 3, 12, 1}) {
        MSELoss(&input, &target);
    }
};

TEST(Model, MSELossTest) {
    const size_t n = 5;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;

    MSETestModel model{};

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> targetData(1 * c * h * w);
    std::vector<float> expectedGrad(n * c * h * w);
    float expectedLoss = 0.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : targetData) {
        i = dist(gen);
    }
    for (auto i = 0; i < inputData.size(); i++) {
        expectedLoss += (inputData[i] - targetData[i % targetData.size()]) * (inputData[i] - targetData[i % targetData.size()]);
    }
    expectedLoss /= static_cast<float>(inputData.size());
    for (auto i = 0; i < inputData.size(); i++) {
        expectedGrad[i] = (inputData[i] - targetData[i % targetData.size()]) * 2.0f / static_cast<float>(inputData.size());
    }

    model.input.dataInject(inputData.begin(), inputData.end());
    model.target.dataInject(targetData.begin(), targetData.end());
    model.forward();
    Tensor expectedGradTensor({n, c, h, w}, true);
    expectedGradTensor.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    expectedGradTensor.dataInject(inputData.begin(), inputData.end());
    model.backward();

    EXPECT_NEAR(expectedLoss, model.getLoss(), 1e-2);
    EXPECT_EQ(expectedGradTensor, *model.input.output);
}

class BCETestModel : public Model {
public:
    InputNode input;
    InputNode target;

    BCETestModel() : input({5, 3, 12, 1}, true), target({1, 3, 12, 1}) {
        BCELoss(&input, &target);
    }
};

TEST(Model, BCELossTest) {
    const size_t n = 5;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;

    BCETestModel model{};

    std::vector<float> inputData(n * c * h * w);
    std::vector<float> targetData(1 * c * h * w);
    std::vector<float> expectedGrad(n * c * h * w);
    float expectedLoss = 0.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto& i : targetData) {
        i = dist(gen);
    }
    for (auto i = 0; i < inputData.size(); i++) {
        expectedLoss += -targetData[i % targetData.size()] * std::log(inputData[i]) - (1 - targetData[i % targetData.size()]) * std::log(1 - inputData[i]);
    }
    expectedLoss /= static_cast<float>(inputData.size());
    for (auto i = 0; i < inputData.size(); i++) {
        expectedGrad[i] = ((inputData[i] - targetData[i % targetData.size()]) / (inputData[i] * (1 - inputData[i]))) / static_cast<float>(inputData.size());
    }

    model.input.dataInject(inputData.begin(), inputData.end());
    model.target.dataInject(targetData.begin(), targetData.end());
    model.forward();
    Tensor expectedGradTensor({n, c, h, w}, true);
    expectedGradTensor.dataInject(expectedGrad.begin(), expectedGrad.end(), true);
    expectedGradTensor.dataInject(inputData.begin(), inputData.end());
    model.backward();

    EXPECT_NEAR(model.getLoss(), expectedLoss, 1e-2);
    EXPECT_EQ(expectedGradTensor, *model.input.output);
}

class SGDTestModel : public Model {
public:
    InputNode input1;
    InputNode input2;

    SGDTestModel() : input1({5, 3, 12, 1}, true), input2({5, 3, 12, 1}) {
        auto x = Add(&input1, &input2);
    }
};

TEST(Model, SGDOptimize) {
    const size_t n = 5;
    const size_t c = 3;
    const size_t h = 12;
    const size_t w = 1;
    const float lr = 0.01f;

    std::vector<float> input1Data(n * c * h * w);
    std::vector<float> input2Data(n * c * h * w);
    std::vector<float> input1Gard(n * c * h * w);
    std::vector<float> expectedData(n * c * h * w);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : input1Data) {
        i = dist(gen);
    }
    for (auto& i : input2Data) {
        i = dist(gen);
    }
    for (auto& i : input1Gard) {
        i = dist(gen);
    }
    for (auto i = 0; i < input1Data.size(); i++) {
        expectedData[i] = input1Data[i] - lr * input1Gard[i];
    }

    SGDTestModel model{};
    model.input1.dataInject(input1Data.begin(), input1Data.end());
    model.input2.dataInject(input2Data.begin(), input2Data.end());
    model.input1.dataInject(input1Gard.begin(), input1Gard.end(), true);

    opt::SGD optimizer(lr);
    model.update(&optimizer);

    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedData.begin(), expectedData.end());
    expected.dataInject(input1Gard.begin(), input1Gard.end(), true);

    EXPECT_EQ(expected, *model.input1.output);
}

TEST(TensorBasic, img2colTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;
    const size_t k_h = 3;
    const size_t k_w = 3;
    const size_t stride = 1;
    const size_t pad = 1;
    const size_t H_out = (h + 2 * pad - k_h) / stride + 1;
    const size_t W_out = (w + 2 * pad - k_w) / stride + 1;

    std::vector<float> inputData({n*c*h*w});
    std::vector<float> expectedData({n*H_out*W_out*k_h*k_w*c});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : inputData) {
        i = dist(gen);
    }

    for (size_t b = 0; b < n; ++b) {
        for (size_t i = 0; i < H_out; ++i) {
            for (size_t j = 0; j < W_out; ++j) {
                const int h_start = static_cast<int>(i * stride) - pad;
                const int w_start = static_cast<int>(j * stride) - pad;

                for (size_t r = 0; r < k_h; ++r) {
                    const int h_in = h_start + r;
                    for (size_t s = 0; s < k_w; ++s) {
                        const int w_in = w_start + s;
                        for (size_t c_in = 0; c_in < c; ++c_in) {
                            float val = 0.0f;
                            if (h_in >= 0 && h_in < h && w_in >= 0 && w_in < w) {
                                const size_t input_idx =
                                    b * (c * h * w) +
                                    c_in * (h * w) +
                                    h_in * w +
                                    w_in;
                                val = inputData[input_idx];
                            }
                            const size_t expected_idx =
                                b * (H_out * W_out * k_h * k_w * c) +
                                (i * W_out + j) * (k_h * k_w * c) +
                                c_in * (k_h * k_w) +
                                r * k_w +
                                s;
                            expectedData[expected_idx] = val;
                        }
                    }
                }
            }
        }
    }

    Tensor input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    auto result = tensorImg2col(input, k_h, k_w, stride, pad);
    Tensor expected({n, 1, H_out * W_out, k_h * k_w * c});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, img2colForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;
    const size_t k_h = 3;
    const size_t k_w = 3;
    const size_t stride = 1;
    const size_t pad = 1;
    const size_t H_out = (h + 2 * pad - k_h) / stride + 1;
    const size_t W_out = (w + 2 * pad - k_w) / stride + 1;

    std::vector<float> inputData({n*c*h*w});
    std::vector<float> expectedData({n*H_out*W_out*k_h*k_w*c});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : inputData) {
        i = dist(gen);
    }

    for (size_t b = 0; b < n; ++b) {
        for (size_t i = 0; i < H_out; ++i) {
            for (size_t j = 0; j < W_out; ++j) {
                const int h_start = static_cast<int>(i * stride) - pad;
                const int w_start = static_cast<int>(j * stride) - pad;

                for (size_t r = 0; r < k_h; ++r) {
                    const int h_in = h_start + r;
                    for (size_t s = 0; s < k_w; ++s) {
                        const int w_in = w_start + s;
                        for (size_t c_in = 0; c_in < c; ++c_in) {
                            float val = 0.0f;
                            if (h_in >= 0 && h_in < h && w_in >= 0 && w_in < w) {
                                const size_t input_idx =
                                    b * (c * h * w) +
                                    c_in * (h * w) +
                                    h_in * w +
                                    w_in;
                                val = inputData[input_idx];
                            }
                            const size_t expected_idx =
                                b * (H_out * W_out * k_h * k_w * c) +
                                (i * W_out + j) * (k_h * k_w * c) +
                                c_in * (k_h * k_w) +
                                r * k_w +
                                s;
                            expectedData[expected_idx] = val;
                        }
                    }
                }
            }
        }
    }

    InputNode input({n, c, h, w});
    input.dataInject(inputData.begin(), inputData.end());
    Img2ColNode result(&input, k_h, k_w, stride, pad);
    result.forward();
    Tensor expected({n, 1, H_out * W_out, k_h * k_w * c});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, img2colBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;
    const size_t k_h = 3;
    const size_t k_w = 3;
    const size_t stride = 1;
    const size_t pad = 1;
    const size_t H_out = (h + 2 * pad - k_h) / stride + 1;
    const size_t W_out = (w + 2 * pad - k_w) / stride + 1;

    std::vector<float> gradData({n*H_out*W_out*k_h*k_w*c});
    std::vector<float> expectedGradData({n*c*h*w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : gradData) {
        i = dist(gen);
    }

    for (size_t b = 0; b < n; ++b) {
        for (size_t i = 0; i < H_out; ++i) {
            for (size_t j = 0; j < W_out; ++j) {
                const int h_start = static_cast<int>(i * stride) - pad;
                const int w_start = static_cast<int>(j * stride) - pad;
                for (size_t r = 0; r < k_h; ++r) {
                    const int h_in = h_start + r;
                    for (size_t s = 0; s < k_w; ++s) {
                        const int w_in = w_start + s;
                        for (size_t c_in = 0; c_in < c; ++c_in) {
                            if (h_in >= 0 && h_in < h && w_in >= 0 && w_in < w) {
                                const size_t input_idx =
                                    b * (c * h * w) +
                                    c_in * (h * w) +
                                    h_in * w +
                                    w_in;
                                const size_t grad_idx =
                                    b * (H_out * W_out * k_h * k_w * c) +
                                    (i * W_out + j) * (k_h * k_w * c) +
                                    c_in * (k_h * k_w) +
                                    r * k_w +
                                    s;
                                expectedGradData[input_idx] += gradData[grad_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    InputNode input({n, c, h, w}, true);
    Img2ColNode result(&input, k_h, k_w, stride, pad);
    result.dataInject(gradData.begin(), gradData.end(), true);
    result.backward();
    Tensor expected({n, c, h, w}, true);
    expected.dataInject(expectedGradData.begin(), expectedGradData.end(), true);
    EXPECT_EQ(expected, *input.output);
}

TEST(TenorBasic, col2imgTest) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> inputData({n*c*h*w});
    std::vector<float> expectedData({n*c*h*w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);

    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            for (auto k = 0; k < h; k++) {
                for (auto l = 0; l < w; l++) {
                    expectedData[i * (c*h*w) + j * (h*w) + k * w + l] =
                        inputData[i * (c*h*w) + (k * w + l) * c + j];
                }
            }
        }
    }

    Tensor input({n ,1, h*w, c});
    input.dataInject(inputData.begin(), inputData.end());
    auto result = tensorCol2img(input, h, w);
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, result);
}

TEST(NodeBasic, col2imgForward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> inputData({n*c*h*w});
    std::vector<float> expectedData({n*c*h*w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            for (auto k = 0; k < h; k++) {
                for (auto l = 0; l < w; l++) {
                    expectedData[i * (c*h*w) + j * (h*w) + k * w + l] =
                        inputData[i * (c*h*w) + (k * w + l) * c + j];
                }
            }
        }
    }

    InputNode input({n ,1, h*w, c});
    input.dataInject(inputData.begin(), inputData.end());
    Col2ImgNode result(&input, h, w);
    result.forward();
    Tensor expected({n, c, h, w});
    expected.dataInject(expectedData.begin(), expectedData.end());
    EXPECT_EQ(expected, *result.output);
}

TEST(NodeBasic, Col2imgBackward) {
    const size_t n = 2;
    const size_t c = 3;
    const size_t h = 4;
    const size_t w = 5;

    std::vector<float> inputData({n*c*h*w});
    std::vector<float> expectedData({n*c*h*w});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.1f, 0.9f);
    for (auto& i : inputData) {
        i = dist(gen);
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < c; j++) {
            for (auto k = 0; k < h; k++) {
                for (auto l = 0; l < w; l++) {
                    expectedData[i * (c*h*w) + j * (h*w) + k * w + l] =
                        inputData[i * (c*h*w) + (k * w + l) * c + j];
                }
            }
        }
    }

    InputNode input({n, 1, h*w, c}, true);
    Col2ImgNode result(&input, h, w);
    result.dataInject(expectedData.begin(), expectedData.end(), true);
    result.backward();
    Tensor expected({n, 1, h*w, c}, true);
    expected.dataInject(inputData.begin(), inputData.end(), true);
    EXPECT_EQ(expected, *input.output);
}