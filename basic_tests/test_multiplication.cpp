
#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(MulTest, testSize) {
    // EX. 1   =======================================================
    matrix ex1_1(3, 3);
    std::vector<std::vector<double>> values1_1 = {{1, 2, 3},
                                                  {4, 5, 9},
                                                  {3, 3, 3}};
    ex1_1 = values1_1;

    matrix ex1_2(3, 3);
    std::vector<std::vector<double>> values1_2 = {{4, 2, 1},
                                                  {4, 8, 3},
                                                  {2, 2, 2}};
    ex1_2 = values1_2;
    auto res1 = (ex1_1.get_cols() == ex1_2.get_rows());

    // EX. 2   =======================================================
    matrix ex2_1(3, 2);
    std::vector<std::vector<double>> values2_1 = {{1, 2},
                                                  {4, 5},
                                                  {3, 3}};
    ex2_1 = values2_1;

    matrix ex2_2(2, 3);
    std::vector<std::vector<double>> values3_2 = {{1, 2, 11},
                                                  {4, 5, 9}};
    ex3_2 = values3_2;

    auto res2 = (ex2_1.get_cols() == ex2_2.get_rows());

    // EX. 3   =======================================================
    matrix ex3_1(2, 3);
    std::vector<std::vector<double>> values3_1 = {{4, 2, 5},
                                                  {4, 8, 2}};
    ex3_1 = values3_1;

    matrix ex3_2(2, 3);
    std::vector<std::vector<double>> values3_2 = {{1, 2, 11},
                                                  {4, 5, 9}};
    ex3_2 = values3_2;

    auto res3 = (ex3_1.get_cols() == ex3_2.get_rows());

    EXPECT_EQ(1, res1);     // square
    EXPECT_EQ(1, res2);     // mxn - nxm
    EXPECT_EQ(0, res3);     // diff
    }

TEST(MulTest, testResult) {
    matrix ex(3, 3);
    std::vector<std::vector<double>> values1_1 = {{1, 2, 3},
                                                  {4, 5, 9},
                                                  {3, 3, 3}};
    ex1_1 = values1_1;

    matrix ex1_2(3, 3);
    std::vector<std::vector<double>> values1_2 = {{4, 2, 1},
                                                  {4, 8, 3},
                                                  {2, 2, 2}};
    ex1_2 = values1_2;

    matrix res2(3, 2);
    res1 = matrix::mul_parallel(ex1_1, ex1_2);
    int res2 = 0;
    if (res1.get_rows() == ex1_1.get_rows() && res1.get_cols() == ex1_2.get_cols()) {res2 = 1}

    ASSERT_EQ(1, res2);
    ASSERT_EQ("{{18,24,13},{54,66,37},{30,36,18}}", res1.to_string_for_test());
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}