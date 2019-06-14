#ifndef PARALLEL_MATRIX_LIBRARY_H
#define PARALLEL_MATRIX_LIBRARY_H

#include <stddef.h>
#include <vector>
#include <sstream>
#include <iostream>

class matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;
public:
    explicit matrix(size_t row_num, size_t col_num, double initial = 0.0) : rows(row_num), cols(col_num) {
        for (size_t i = 0; i < row_num; ++i) data.emplace_back(col_num, initial);

    }

    size_t get_cols() const { return cols; }

    size_t get_rows() const { return rows; }

    double &operator()(size_t i, size_t j) {
        return data[i][j];
    }

    const double &operator()(size_t i, size_t j) const {
        return data[i][j];
    }

    std::string to_string() {
        std::ostringstream result;
        for (auto &v: data) {
            for (auto &i: v) {
                result << i << "\t\t";
            }
            result << "\n";
        }
        return result.str();
    }

    std::string to_string_for_test() {
        std::ostringstream result;
        result << '{';
        for (auto v = data.begin(); v!=data.end(); v++) {
            if (*data.begin() != (*v)) result << ",";
            result << '{';
            for (auto i = (*v).begin();i != (*v).end(); ++i) {
                if ((*v).begin() != i){
                    result << ",";
                }
                result << *i;
            }
            result << "}";
        }
        result << '}';

        return result.str();
    }

    std::vector<std::vector<double>> subcols(size_t from_col, size_t to_col)
    {

        std::vector<std::vector<double>> m_row;
        std::vector<std::vector<double>> res;

        for (auto row: data){
            row.erase(row.begin(), row.begin() + from_col);
            row.erase(row.begin() + to_col - from_col, row.end());
            m_row.push_back(row);
        }

        for ( size_t i = 0; i < m_row[0].size(); ++i) {
            std::vector<double> vect;
            res.push_back(vect);
        }

        for ( size_t i = 0; i < m_row[0].size(); ++i) {
            for (size_t j = 0; j < m_row.size(); ++j) {
                res[i].push_back(m_row[j][i]);
            }
        }

        return res;
    }


    matrix del_rows(size_t from_row, size_t to_row)
    {
        data.erase(data.begin() + from_row, data.begin() + to_row);
        rows -= to_row - from_row;
        return *this;
    }


    matrix del_cols(size_t from_col, size_t to_col)
    {
        for (auto &row: data){
            row.erase(row.begin() + from_col, row.begin() + to_col);
        }
        cols-= to_col - from_col;
        return *this;
    }


    matrix &operator=(const std::vector<std::vector<double>> &init_values);

    static matrix add(const matrix &m1, const matrix &m2);

    static matrix sub(const matrix &m1, const matrix &m2);

    static matrix mul(const matrix &m1, const matrix &m2);

    static matrix inverse(const matrix &m1);

    static matrix inverse_parallel(const matrix &m1);

    static matrix sub_parallel(const matrix &m1, const matrix &m2);

    static matrix add_parallel(const matrix &m1, const matrix &m2);

    static matrix mul_parallel(const matrix &m1, const matrix &m2);

    static matrix mul_on_num(const matrix &m1, double num);

    static matrix eigenvectors(matrix &m, std::vector<double> &eigenvalues);

    static std::vector<double> eigenvalues(const matrix &m1);

    static matrix eigenvectors_parallel(matrix &m, std::vector<double> &eigenvalues);

    static std::vector<double> eigenvalues_parallel(matrix &m1);

    };


#endif // PARALLEL_MATRIX_LIBRARY_H
