#include <iostream>
#include "matrix.hpp"
#include <fstream>

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
//    // Some examples of usage
//    // For more detailed instructions open README.md file

    std::string name = "../matrices.txt";
    matrix large_m(150, 150);
    large_m = read_from_file(name, 150,  150);

    matrix M_1(3, 3);
    std::vector<std::vector<double>> values_1 = {{1, 2, 3},
                                                 {4, 5, 4},
                                                 {1, 1, 1}};

    M_1 = values_1;
    matrix M_2(3, 3);
    std::vector<std::vector<double>> values_2 = {{-3, 1, -3},
                                                 {2,5,-3},
                                                 {5,3,-3}};

    std::vector<double> evalues = {-0.822, 0.342, 7.498};
    auto e = matrix::eigenvectors(M_1, evalues);
    std::cout << e.to_string() << std::endl;

    matrix M_inverse(150, 150);
    M_inverse = matrix::inverse_parallel(large_m);
    std::cout << M_inverse.to_string() << "\n";

    return 0;

}
