#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(AddTest, sameSize) {

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
    auto res1 = (ex1_1.get_rows() == ex1_2.get_rows() && ex1_1.get_cols() == ex1_2.get_cols());

    // EX. 2   =======================================================
    matrix ex2_1(3, 2);
    std::vector<std::vector<double>> values2_1 = {{1, 2},
                                                  {4, 5},
                                                  {3, 3}};
    ex2_1 = values2_1;

    matrix ex2_2(3, 2);
    std::vector<std::vector<double>> values2_2 = {{4, 2},
                                                  {4, 8},
                                                  {2, 2}};
    ex2_2 = values2_2;
    auto res2 = (ex2_1.get_rows() == ex2_2.get_rows() && ex2_1.get_cols() == ex2_2.get_cols());

    // EX. 3   =======================================================
    matrix ex3_1(3, 3);
    std::vector<std::vector<double>> values3_1 = {{1, 2, 3},
                                                  {4, 5, 9},
                                                  {3, 3, 3}};
    ex3_1 = values3_1;

    matrix ex3_2(3, 2);
    std::vector<std::vector<double>> values3_2 = {{4, 2},
                                                  {4, 8},
                                                  {2, 2}};
    ex3_2 = values3_2;
    auto res3 = (ex3_1.get_rows() == ex3_2.get_rows() && ex3_1.get_cols() == ex3_2.get_cols( ));

    // TESTS
    EXPECT_EQ(1, res1);     // square
    EXPECT_EQ(1, res2);     // rect
    EXPECT_EQ(0, res3);     // diff
}

TEST(AddTest, sameResult) {
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

    matrix res1(3, 3);
    res1 = matrix::add_parallel(ex1_1, ex1_2);

    // EX. 2   =======================================================
    matrix ex2_1(3, 2);
    std::vector<std::vector<double>> values2_1 = {{1, 2},
                                                  {4, 5},
                                                  {3, 3}};
    ex2_1 = values2_1;

    matrix ex2_2(3, 2);
    std::vector<std::vector<double>> values2_2 = {{4, 2},
                                                  {4, 8},
                                                  {2, 2}};
    ex3_2 = values2_2;

    matrix res2(3, 2);
    res2 = matrix::add_parallel(ex2_1, ex2_2);

    // EX. 3   =======================================================
    matrix ex3_1(2, 3);
    std::vector<std::vector<double>> values3_1 = {{4, 2, 5},
                                                  {4, 8, 2}};
    ex3_1 = values3_1;

    matrix ex3_2(2, 3);
    std::vector<std::vector<double>> values3_2 = {{1, 2, 11},
                                                  {4, 5, 9}};
    ex3_2 = values3_2;

    matrix res3(3, 2);
    res3 = matrix::add_parallel(ex3_1, ex3_2);

    EXPECT_EQ("{{5,4,4},{8,13,12},{5,5,5}}", res1.to_string_for_test());    // square
    EXPECT_EQ("{{5,4},{8,13}, {5,5}}", res2.to_string_for_test());             // rect
    EXPECT_EQ("{{5,4,16},{8,13,11}}", res3.to_string_for_test());             // inv rect
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}