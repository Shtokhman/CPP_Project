#include <iostream>
#include "matrix.hpp"
#include <fstream>
#include <chrono>
#include <atomic>

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

template<class D>
inline long long to_us(const D& d)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}


std::vector<std::string> split(const std::string& str)
{
    std::vector<std::string> cont;
    std::istringstream iss(str);
    std::copy(std::istream_iterator<std::string>(iss),
              std::istream_iterator<std::string>(),
              std::back_inserter(cont));
    return cont;
}

std::vector<std::vector<double>> read_from_file(std::string &file_name, size_t num_cols, size_t num_rows){
    std::ifstream in(file_name);
    auto ss = std::ostringstream{};
    ss << in.rdbuf();
    std::string str = ss.str();

    in.close();
    std::vector<std::string> splited_str = split(str);


    std::vector<std::vector<double> > res;
    int n_c = 0;
    int n_r = 0;

    try {
        std::vector<double> vect;
        for (const auto &i: splited_str) {
            double num = std::stod(i);
            vect.push_back(num);
            if (n_c == num_cols -1 ){
                res.push_back(vect);
                vect.clear();
                n_c = 0;
                if (n_r == num_rows-1) break;
                n_r++;
            }
            n_c++;

        }
    }

    catch(std::invalid_argument &e)
    {
        std::cerr << "Error, not correct configuration" << std::endl;
    }

    return res;
}



int main() {
    std::string name = "../matrices.txt";
    matrix large_m(400, 400);
    large_m = read_from_file(name, 400,  400);


    matrix m_1(3, 3);
    std::vector<std::vector<double>> values_1 = {{1, 2, 3},
                                                 {4, 5, 6},
                                                 {7, 8, 1}};
    m_1 = values_1;

    matrix m_2(6, 6);
    std::vector<std::vector<double>> values_2 = {{2,0,0,0,0,1},
                                                 {0,0,4,0,0,0},
                                                  {0,0,0,0,9,0},
                                                  {0,10,0,0,0,0},
                                                  {0,0,0,3,3,0},
                                                  {0,0,0,0,7,0}};
    m_2 = values_2;


    // testing result;

    std::cout << matrix::add(m_1, m_1).to_string() << std::endl;
    std::cout << matrix::add_parallel(m_1, m_1).to_string() << std::endl;

    std::cout << matrix::sub(m_1, m_1).to_string() << std::endl;
    std::cout << matrix::sub_parallel(m_1, m_1).to_string() << std::endl;

    std::cout << matrix::mul(m_1, m_1).to_string() << std::endl;
    std::cout << matrix::mul_parallel(m_1, m_1).to_string() << std::endl;

    std::cout << matrix::inverse(m_1).to_string() << std::endl;
    std::cout << matrix::inverse_parallel(m_1).to_string() << std::endl;

    auto e_res1 = matrix::eigenvalues(m_2);
    auto e_res2 = matrix::eigenvalues(m_2);

    matrix e1(1, e_res1.size());
    e1 = {e_res1};
    std::cout << e1.to_string() << std::endl;
    e1 = {e_res2};
    std::cout << e1.to_string() << std::endl;

    std::vector<double> eigenvalues = {-5.074, -0.380, 12.454};

    std::cout << matrix::eigenvectors(m_1, eigenvalues).to_string() << std::endl;
    std::cout << matrix::eigenvectors_parallel(m_1, eigenvalues).to_string() << std::endl;

    auto s_add_p = get_current_time_fenced();
    auto M_res_p = matrix::mul_parallel(large_m, large_m);
    auto f_add_p = get_current_time_fenced();

    auto s_add = get_current_time_fenced();
    auto M_res = matrix::mul(large_m, large_m);
    auto f_add = get_current_time_fenced();

    std::cout << "mul parallel: " << to_us(f_add_p - s_add_p)<< std::endl;
    std::cout << "mul seq: " << to_us(f_add - s_add)<< std::endl;

    return 0;

}
