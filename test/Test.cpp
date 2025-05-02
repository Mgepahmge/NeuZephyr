#include <gtest/gtest.h>
#include <random>

#include <TensorOperations.cuh>
#include <Nodes.cuh>
using namespace nz::data;
using namespace nz::nodes;
using namespace nz::nodes::calc;
using namespace nz::nodes::io;
using namespace nz::nodes::loss;
using namespace nz;

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
