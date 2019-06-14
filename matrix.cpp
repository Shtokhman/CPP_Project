#include "matrix.hpp"
#include <thread>
#include <math.h>

std::vector<size_t> make_vector_steps_for_threads(size_t num, size_t &num_threads){
    std::vector<size_t> steps;
    if (num_threads > num) num_threads = num;
    int step = static_cast<int>(num/num_threads);
    for (size_t i = 0; i < num_threads; i++) steps.push_back(i*step);
    steps.push_back(num);
    return steps;
}


void subtract_matrix(matrix &addition_matrix, const matrix &m1, const matrix &m2, size_t from, size_t to) {
    for (size_t i = from; i < to; i++) {
        for (size_t j = 0; j < m1.get_cols(); j++) {
            addition_matrix(i, j) = m1(i, j) - m2(i, j);
        }
    }
}

void add_matrix(matrix &addition_matrix, const matrix &m1, const matrix &m2, size_t from, size_t to) {
    for (size_t i = from; i < to; i++) {
        for (size_t j = 0; j < m1.get_cols(); j++) {
            addition_matrix(i, j) = m1(i, j) + m2(i, j);
        }
    }
}

void mul_matrix(matrix &mull_matrix, const matrix &m1, const matrix &m2, size_t from, size_t to){
    double item = 0;
    for (size_t i = from; i < to; i++) {
        for (size_t j = 0; j < m2.get_cols(); j++) {
            for (size_t k = 0; k < m2.get_rows(); k++) {
                item += m1(i, k) * m2(k, j);
            }
            mull_matrix(i, j) = item;
            item = 0;
        }
    }
}

matrix matrix::sub_parallel(const matrix &m1, const matrix &m2) {
    size_t num_threads = std::thread::hardware_concurrency();
    if (m1.get_cols() != m2.get_cols() || m1.get_rows() != m2.get_rows()) {
        std::runtime_error("matrices are of different sizes");
    }
    matrix addition_matrix(m1.get_rows(), m1.get_cols());
    std::vector<size_t> steps = make_vector_steps_for_threads(m1.get_rows(), num_threads);
    std::thread myThreads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        myThreads[i] = std::thread(subtract_matrix, std::ref(addition_matrix), std::ref(m1), std::ref(m2), steps[i], steps[i + 1]);
    }
    for (int i = 0; i < num_threads; i++) myThreads[i].join();

    return addition_matrix;

}

matrix matrix::add_parallel(const matrix &m1, const matrix &m2) {
    size_t num_threads = std::thread::hardware_concurrency();
    if (m1.get_cols() != m2.get_cols() || m1.get_rows() != m2.get_rows()) {
        std::runtime_error("matrices are of different sizes");
    }
    matrix addition_matrix(m1.get_rows(), m1.get_cols());
    std::vector<size_t> steps = make_vector_steps_for_threads(m1.get_rows(), num_threads);
    std::thread myThreads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        myThreads[i] = std::thread(add_matrix, std::ref(addition_matrix), std::ref(m1), std::ref(m2), steps[i], steps[i + 1]);
    }
    for (int i = 0; i < num_threads; i++) myThreads[i].join();

    return addition_matrix;

}

matrix matrix::mul_parallel(const matrix &m1, const matrix &m2) {
    size_t num_threads = std::thread::hardware_concurrency();
    if (m1.get_cols() != m2.get_rows() || m1.get_rows() != m2.get_cols()) {
        std::runtime_error("matrices are of different sizes");
    }

    matrix multiplication_matrix(m1.get_rows(), m1.get_rows());
    std::vector<size_t> steps = make_vector_steps_for_threads(m1.get_rows(), num_threads);
    std::thread myThreads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        myThreads[i] = std::thread(mul_matrix, std::ref(multiplication_matrix), std::ref(m1), std::ref(m2), steps[i],
                                   steps[i + 1]);
    }

    for (int i = 0; i < num_threads; i++) myThreads[i].join();

    return multiplication_matrix;
}

matrix matrix::add(const matrix &m1, const matrix &m2) {
    if (m1.get_cols() != m2.get_cols() || m1.get_rows() != m2.get_rows()) {
        std::runtime_error("matrices are of different sizes");
    }
    matrix addition_matrix(m1.get_rows(), m1.get_cols());
    for (size_t i = 0; i < m1.get_rows(); i++) {
        for (size_t j = 0; j < m1.get_cols(); j++) {
            addition_matrix(i, j) = m1(i, j) + m2(i, j);
        }
    }
    return addition_matrix;
}

matrix matrix::sub(const matrix &m1, const matrix &m2) {
    // check matrix sizes
    if (m1.get_cols() != m2.get_cols() || m1.get_rows() != m2.get_rows()) {
        std::runtime_error("matrices are of different sizes");
    }
    matrix addition_matrix(m1.get_rows(), m1.get_cols());

    for (size_t i = 0; i < m1.get_rows(); i++) {
        for (size_t j = 0; j < m1.get_cols(); j++) {
            addition_matrix(i, j) = m1(i, j) - m2(i, j);
        }
    }
    return addition_matrix;
}

matrix matrix::mul(const matrix &m1,const matrix &m2) {
    if (m1.get_cols() != m2.get_rows() || m1.get_rows() != m2.get_cols()) {
        std::runtime_error("matrices are of different sizes");
    }

    matrix multiplication_matrix(m1.get_rows(), m1.get_rows());

    for (size_t i = 0; i < m1.get_rows(); i++) {
        for (size_t j = 0; j < m2.get_cols(); j++) {
            for (size_t k = 0; k < m2.get_rows(); k++) {
                multiplication_matrix(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }
    return multiplication_matrix;
}

matrix swap(matrix &m, size_t row1, size_t row2, size_t col)
{
    for (size_t i = 0; i < col; i++)
    {
        double temp = m(row1 ,i);
        m(row1, i) = m(row2, i);
        m(row2, i) = temp;
    }
    return m;
}


bool not_equal(double a, double b, double epsilon){
    return abs(a-b) > epsilon;
}

size_t rankOfMatrix(matrix m)
{
    size_t R = m.get_rows();
    size_t rank = m.get_cols();

    for (size_t row = 0; row < rank; row++)
    {
        if (not_equal(m(row, row), 0, pow(10, -5)))
        {
            for (size_t col = 0; col < R; col++)
            {
                if (col != row)
                {
                    double mult = m(col, row)/m(row, row);
                    for (size_t i = 0; i < rank; i++) {
                        m(col, i) -= mult * m(row, i);
                    }
                }
            }
        }

        else
        {
            bool reduce = true;
            for (size_t i = row + 1; i < R; i++)
            {
                if (not_equal(m(i, row), 0, pow(10, -5)))
                {
                    swap(m, row, i, rank);
                    reduce = false;
                    break ;
                }
            }
            if (reduce)
            {
                rank--;
                for (size_t i = 0; i < R; i ++)
                    m(i, row) = m(i, rank);
            }
            row--;
        }
    }
    return rank;
}

void LU_fact(const matrix &m1, matrix &U, matrix &L, matrix &identity){
    size_t dim = m1.get_rows();
    for (size_t i = 0; i < dim; ++i) {
        for (size_t k = i; k < dim; ++k) {
            double sum = 0;
            for (size_t j = 0; j < i; ++j) {
                sum += (L(i, j) * U(j, k));
            }
            U(i, k) = m1(i, k) - sum;
        }

        for (size_t k = i; k < dim; ++k) {
            if (i == k) {
                L(i, i) = 1;
            } else {
                double sum = 0;
                for (size_t j = 0; j < i; ++j) {
                    sum += (L(k, j) * U(j, i));
                }
                L(k, i) = (m1(k, i) - sum) / U(i, i);
            }
        }
    }
}

void transpose(matrix &transpose, matrix &m){
    size_t dim = m.get_rows();
    for (size_t j = 0; j < dim; ++j) {
        for (size_t i = 0; i < dim; ++i) {
            transpose(j, i) = m(i, j);
        }
    }
}

matrix identety_matrix(size_t size) {
    matrix I(size, size);
    for (size_t i = 0; i < size; ++i)
        I(i,i) = 1;
    return I;
}

void invert_matrix(matrix &prefinal, size_t &dim, matrix &identity, matrix &L, matrix &U)
{
    double z[dim];
    double b[dim];
    double sum;
    for (size_t c = 0; c < dim; ++c) {
        for (size_t i = 0; i < dim; ++i) {

            sum = identity(c, i);
            for (size_t j = 0; j < i; ++j) {
                sum -= L(i, j) * z[j];
            }
            z[i] = sum / L(i, i);
        }
        for (int i = dim-1; i >= 0; --i) {
            sum = z[i];
            for (size_t j = i+1; j < dim; ++j) {
                sum = sum - U(i, j)*b[j];
            }
            b[i] = sum / U(i, i);
            prefinal(c, i) = b[i];

        }
    }
}

matrix matrix::inverse(const matrix &m1) {
    if (m1.get_cols() != m1.get_rows()) {
        std::runtime_error("matrix is not square");
    }

    size_t dim = m1.get_cols();
    matrix inverted(dim, dim);
    if (rankOfMatrix(m1) != m1.get_cols())
    {
        std::cout << "Matrix are non-invertible" << std::endl;
        return inverted;
    }

    matrix L(dim, dim);
    matrix U(dim, dim);
    matrix identity = identety_matrix(dim);
    matrix prefinal(dim, dim);

    LU_fact(m1, U, L, identity);
    invert_matrix(prefinal, dim, identity, L, U);
    transpose(inverted, prefinal);

    return inverted;
}

void transpose_parallel(matrix my_matrix, matrix &transpose, size_t dim, size_t from, size_t to){
    for (size_t j = from; j < to; ++j) {
        for (size_t i = 0; i < dim; ++i) {
            transpose(j, i) = my_matrix(i, j);
        }
    }
}

matrix matrix::inverse_parallel(const matrix &m1) {
    if (m1.get_cols() != m1.get_rows()) {
        std::runtime_error("matrix is not square");
    }

    size_t dim = m1.get_cols();
    matrix inverted(dim, dim);
    if (rankOfMatrix(m1) != m1.get_cols())
    {
        std::cout << "Matrix are non-invertible" << std::endl;
        return inverted;
    }

    matrix L(dim, dim);
    matrix U(dim, dim);
    matrix identity = identety_matrix(dim);
    matrix prefinal(dim, dim);

    LU_fact(m1, U, L, identity);
    invert_matrix(prefinal, dim, identity, L, U);

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<size_t> steps = make_vector_steps_for_threads(dim, num_threads);
    std::thread my_Threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        my_Threads[i] = std::thread(transpose_parallel, std::ref(prefinal), std::ref(inverted), m1.get_rows(), steps[i] ,steps[i+1]);
    }
    for (int i = 0; i < num_threads; i++) my_Threads[i].join();
    return inverted;
}



matrix new_m_transpose(const matrix &m1){
    matrix transpose(m1.get_cols(), m1.get_rows());
    for (size_t i = 0; i < m1.get_rows(); ++i) {
        for (size_t j = 0; j < m1.get_cols(); ++j) {
            transpose(j, i) = m1(i, j);
        }
    }
    return transpose;
}


double vector_distance(std::vector<double> const &vect)
{
    double res = 0;
    for(auto &x: vect)
        res += pow(x, 2);
    return sqrt(res);
}

std::vector<double> e_vector(size_t size)
{
    std::vector<double> e_vector = {1};
    for(size_t i = 1; i < size; ++i)
        e_vector.push_back(0);
    return e_vector;
}


matrix householder(const matrix &v){
    matrix I = identety_matrix(v.get_cols());
    matrix m1 = matrix::mul(new_m_transpose(v), v);
    auto m2 = matrix::mul_on_num(m1, 2);
    return matrix::sub(I, m2);;
}

matrix vector_for_householder(std::vector<double> &col, size_t col_num, double a_1)
{
    int sign_a_1 = 1;
    if (a_1 < 0) sign_a_1 = -1;
    matrix v_1(1, col_num);
    matrix e(1, col_num);

    auto dist_col_i = vector_distance(col);
    e = {e_vector(col_num)};
    v_1 = {col};
    auto v_2 = matrix::mul_on_num(e, sign_a_1*dist_col_i);
    return matrix::add(v_1, v_2);
}

matrix matrix_for_householder(const matrix &cur_m, size_t full_col_num)
{
    auto diff = full_col_num - cur_m.get_cols();
    auto res = identety_matrix(full_col_num);
    for (size_t i = 0; i < cur_m.get_rows(); ++i)
    {
        for (size_t j = 0; j < cur_m.get_cols() ; ++j)
            res(i + diff, j + diff) = cur_m(i, j);
    }
    return res;
}

matrix q_from_householder_reflections(const matrix &m)
{
    matrix Q(m.get_cols(), m.get_cols());
    auto a_1 = m(0 ,0);
    auto H_M = m;
    auto previous_H_M = m;
    for (size_t i = 0; i < m.get_cols(); i++)
    {
        auto col_i = H_M.subcols(0, 1);
        auto v = vector_for_householder(col_i[0], H_M.get_cols(), a_1);
        auto H = matrix_for_householder(householder(v), m.get_cols());
        if (i == 0) Q = H;
        else {Q = matrix::mul(Q, H);};
        H_M = matrix::mul(H, previous_H_M);
        previous_H_M = H_M;
        H_M.del_rows(0, i+1);
        H_M.del_cols(0, i+1);
    }
    return Q;
}

matrix matrix_correspond_eigenvalue(const matrix &m, const double eigenvalue)
{
    auto result = m;
    for (size_t i = 0; i < m.get_rows(); i++) {
        result(i, i) = m(i, i) - eigenvalue;
    }
    return result;
}

std::vector<double> calculate_eigenvector(double value, matrix &m)
{
    auto correspond_to_eigenvalue = matrix_correspond_eigenvalue(m, value);
    size_t rank = m.get_cols() - 1;
    auto tr = new_m_transpose(correspond_to_eigenvalue);
    auto q = q_from_householder_reflections(tr);
    std::vector<std::vector<double>> eigenvector = q.subcols(rank, rank+1);
    return eigenvector[0];
}

matrix matrix::eigenvectors(matrix &m, std::vector<double> &eigenvalues)
{
    std::vector<std::vector<double>> eigenvectors;
    if (m.get_rows() != m.get_cols()) {
        std::runtime_error("matrices are not square");
    }

    eigenvectors.reserve(eigenvalues.size());
    for (auto &value: eigenvalues)
        eigenvectors.push_back(calculate_eigenvector(value, m));

    matrix res(eigenvalues.size(), m.get_rows());
    res = eigenvectors;
    return res;
}

void calculate_eigenvector_for_parallel(std::vector<std::vector<double>> &eigenvectors, std::vector<double> &values, matrix &m, size_t from, size_t to, std::mutex &my_mutex)
{
    for (size_t i = from; i < to; ++i)
    {
        auto vector = calculate_eigenvector(values[i], m);
        my_mutex.lock();
        eigenvectors.push_back(vector);
        my_mutex.unlock();
    }
}

matrix matrix::mul_on_num(const matrix &m1, double num)
{
    matrix mul_matrix(m1.get_rows(), m1.get_cols());
    for (size_t i = 0; i < m1.get_rows(); i++) {
        for (size_t j = 0; j < m1.get_cols(); j++) {
            mul_matrix(i, j) = m1(i, j) * num;
        }
    }
    return mul_matrix;
}

matrix matrix::eigenvectors_parallel(matrix &m, std::vector<double> &eigenvalues)
{
    std::mutex my_mutex;
    std::vector<std::vector<double>> eigenvectors;
    if (m.get_rows() != m.get_cols()) {
        std::runtime_error("matrices are not square");
    }
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<size_t> steps = make_vector_steps_for_threads(eigenvalues.size(), num_threads);
    std::thread myThreads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        myThreads[i] = std::thread(calculate_eigenvector_for_parallel, std::ref(eigenvectors), std::ref(eigenvalues), std::ref(m), steps[i], steps[i+1], std::ref(my_mutex));
    }
    for (int i = 0; i < num_threads; i++) myThreads[i].join();
    matrix res(eigenvalues.size(), m.get_rows());
    res = eigenvectors;
    return res;
}


std::vector<double> matrix::eigenvalues(const matrix &m1) {
    std::vector<double> list(m1.get_rows());
    std::vector<double> eigenvalues(m1.get_rows());
    if (m1.get_rows() != m1.get_cols()) {
        std::runtime_error("matrices are of different sizes");
    }

    for (size_t i = 0; i < m1.get_rows(); i++)
        list[i] = 1;

    for (size_t i = 0; i < m1.get_rows(); i++) {
        for (size_t j = 0; j < m1.get_rows(); j++)
            eigenvalues[i] += + m1(i,j) * list[j];
    }
    return eigenvalues;
}

void eigenvalues_for_parallel(std::vector<double> &eigenvalues, matrix &m1, std::vector<double> &list, size_t from, size_t to)
{
    for (size_t i = from; i < to; i++) {
        for (size_t j = 0; j < m1.get_rows(); j++)
            eigenvalues[i] += + m1(i,j) * list[j];
    }
}

std::vector<double> matrix::eigenvalues_parallel(matrix &m1) {
    std::vector<double> list(m1.get_rows());
    std::vector<double> eigenvalueslst(m1.get_rows());
    if (m1.get_rows() != m1.get_cols()) {
        std::runtime_error("matrices are of different sizes");
    }
    for (size_t i = 0; i < m1.get_rows(); i++)
        list[i] = 1;

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<size_t> steps = make_vector_steps_for_threads(m1.get_rows(), num_threads);
    std::thread myThreads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        myThreads[i] = std::thread(eigenvalues_for_parallel, std::ref(eigenvalueslst), std::ref(m1), std::ref(list), steps[i], steps[i+1]);
    }
    for (int i = 0; i < num_threads; i++) myThreads[i].join();
    return eigenvalueslst;
}

matrix &matrix::operator=(const std::vector<std::vector<double>> &init_values) {
    if (init_values.size() != rows || init_values[0].size() != cols) {
        std::runtime_error("size of init 2d array doesn't match matrix dimensions");
    }
    for (size_t i = 0; i < init_values.size(); ++i) {
        for (size_t j = 0; j < init_values[i].size(); ++j) {
            data[i][j] = init_values[i][j];
        }
    }
    return *this;
}
