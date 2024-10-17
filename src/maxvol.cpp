#include "maxvol.h"
#include <iostream>

using namespace Eigen;

std::pair<VectorXi, MatrixXd> maxvol(const MatrixXd& A, double e, int k) {
    int n = A.rows();
    int r = A.cols();

    if (n <= r) {
        throw std::invalid_argument("Input matrix should be 'tall'");
    }

    FullPivLU<MatrixXd> lu_decomp(A);
    VectorXi I = lu_decomp.permutationP().indices().head(r);
    MatrixXd Q = lu_decomp.matrixLU().triangularView<Upper>().solve(A.transpose());
    MatrixXd B = lu_decomp.matrixLU().triangularView<UnitLower>().solve(Q).transpose();

    for (int iter = 0; iter < k; ++iter) {
        int i, j;
        B.cwiseAbs().maxCoeff(&i, &j);
        if (std::abs(B(i, j)) <= e) {
            break;
        }

        I(j) = i;

        VectorXd bj = B.col(j);
        VectorXd bi = B.row(i);
        bi(j) -= 1.0;

        B -= bj * bi.transpose() / B(i, j);
    }

    return {I, B};
}

std::pair<VectorXi, MatrixXd> maxvol_rect(const MatrixXd& A, double e, int dr_min, int dr_max, double e0, int k0) {
    int n = A.rows();
    int r = A.cols();
    int r_min = r + dr_min;
    int r_max = dr_max == -1 ? n : std::min(r + dr_max, n);

    if (r_min < r || r_min > r_max || r_max > n) {
        throw std::invalid_argument("Invalid minimum/maximum number of added rows");
    }

    auto [I0, B] = maxvol(A, e0, k0);

    VectorXi I = VectorXi::Zero(r_max);
    I.head(r) = I0;
    VectorXi S = VectorXi::Ones(n);
    for (int idx = 0; idx < I0.size(); ++idx) {
        S(I0(idx)) = 0;  // Set the corresponding elements to zero
    }
    VectorXd F = S.cast<double>().array() * B.rowwise().norm().array().square();

    for (int k = r; k < r_max; ++k) {
        int i;
        F.maxCoeff(&i);

        if (k >= r_min && F(i) <= e * e) {
            break;
        }

        I(k) = i;
        S(i) = 0;

        VectorXd v = B * B.row(i).transpose();
        double l = 1.0 / (1.0 + v(i));
        B = B - l * v * B.row(i);
        B.conservativeResize(B.rows(), B.cols() + 1);
        B.col(B.cols() - 1) = l * v;

        F = S.cast<double>().array() * (F.array() - l * v.array().square());
    }

    I.conservativeResize(B.cols());
    B(I) = MatrixXd::Identity(B.cols(), B.cols());

    return {I, B};
}