#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(invTest, squareSize) {
    // EX. 1   =======================================================
    matrix ex1_1(3, 3);
    std::vector<std::vector<double>> values1_1 = {{1, 2, 3},
                                                  {4, 5, 9},
                                                  {3, 3, 3}};
    ex1_1 = values1_1;
    auto res1 = (ex1_1.get_rows() == ex1_1.get_cols());

    // EX. 2   =======================================================
    matrix ex2_1(3, 2);
    std::vector<std::vector<double>> values2_1 = {{1, 2},
                                                  {4, 5},
                                                  {3, 3}};
    ex2_1 = values2_1;

    auto res2 = (ex2_1.get_rows == ex2_1.get_cols);


    EXPECT_EQ(1, res1);     // square
    EXPECT_EQ(0, res2);     // rect
    }

TEST(invTest, sameResult) {
    // EX. 1   =======================================================
    matrix ex1_1(3, 3);
    std::vector<std::vector<double>> values1_1 = {{1, 2, 3},
                                                  {4, 5, 9},
                                                  {3, 3, 3}};
    ex1_1 = values1_1;

    matrix res1(3, 3);
    res1 = matrix::inverse(ex1_1);

    EXPECT_EQ("{{-1.33333, 0.333333, 0.333333}, {1.66667, -0.666667, 0.333333}, {-0.333333, 0.333333, -0.333333}}", res1.);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}