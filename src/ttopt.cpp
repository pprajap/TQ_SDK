#include "ttopt.h"
#include <random>
#include <iostream>

using namespace Eigen;

std::pair<VectorXi, double> ttopt(
    std::function<std::pair<VectorXd, VectorXd>(const MatrixXi&, const VectorXi&, double, double)> f,
    const std::vector<int>& n, int rank, int evals, 
    const std::vector<MatrixXd>& Y0, int seed, double fs_opt,
    bool add_opt_inner, bool add_opt_outer, bool add_opt_rect,
    bool add_rnd_inner, bool add_rnd_outer, 
    const std::vector<MatrixXi>& J0, bool is_max) {

    int d = n.size();
    evals = evals == -1 ? std::numeric_limits<int>::max() : evals;

    std::vector<MatrixXi> J_list(d + 1);
    std::vector<int> r(d + 1, 1);

    if (J0.empty()) {
        auto [Y0_init, r_init] = ttopt_init(n, rank, Y0, seed, true);
        J_list = std::vector<MatrixXi>(d + 1, MatrixXi());
        for (int i = 0; i < d - 1; ++i) {
            // Implement _iter function
        }
    } else {
        J_list = J0;
        for (int i = 1; i < d; ++i) {
            r[i] = std::min(rank, n[i - 1] * r[i - 1]);
        }
    }

    VectorXi i_opt;
    double y_opt = is_max ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    double opt_opt = 0.0;

    int eval = 0;
    int iter = 0;
    int i = d - 1;
    bool l2r = false;

    while (true) {
        // Implement _merge function
        MatrixXi I; // = _merge(J_list[i], J_list[i + 1], Jg_list[i]);

        int eval_curr = I.rows();
        if (eval + eval_curr > evals) {
            I.conservativeResize(evals - eval, I.cols());
        }

        auto [y, opt] = f(I, i_opt, y_opt, opt_opt);

        if (y.size() == 0) {
            return {i_opt, y_opt};
        }

        std::tie(i_opt, y_opt) = ttopt_find(I, y, opt, i_opt, y_opt, opt_opt, is_max);

        eval += y.size();
        if (eval >= evals) {
            return {i_opt, y_opt};
        }

        if (y.size() < I.rows()) {
            return {i_opt, y_opt};
        }

        MatrixXd Z = Map<MatrixXd>(y.data(), r[i], n[i] * r[i + 1]);
        if (!is_max) {
            Z = ttopt_fs(Z, y_opt, fs_opt);
        }

        // Implement _iter and _update_iter functions

        // Update the current core index
        // std::tie(i, iter, l2r) = _update_iter(d, i, iter, l2r);
    }

    return {i_opt, y_opt};
}

std::pair<VectorXi, double> ttopt_find(
    const MatrixXi& I, const VectorXd& y, const VectorXd& opt, 
    const VectorXi& i_opt, double y_opt, double opt_opt, bool is_max) {

    int ind = is_max ? y.maxCoeff() : y.minCoeff();
    double y_opt_curr = y[ind];

    if ((is_max && y_opt_curr <= y_opt) || (!is_max && y_opt_curr >= y_opt)) {
        return {i_opt, y_opt};
    }

    return {I.row(ind), y_opt_curr};
}

MatrixXd ttopt_fs(const MatrixXd& y, double y0, double opt) {
    if (opt == 0) {
        return M_PI / 2 - y.array().atan();
    } else {
        return (-opt * (y.array() - y0)).exp();
    }
}

std::pair<std::vector<MatrixXd>, std::vector<int>> ttopt_init(
    const std::vector<int>& n, int rank, const std::vector<MatrixXd>& Y0, 
    int seed, bool with_rank) {

    int d = n.size();
    std::vector<int> r(d + 1, 1);

    for (int i = 1; i < d; ++i) {
        r[i] = std::min(rank, n[i - 1] * r[i - 1]);
    }

    std::mt19937 rng(seed);
    std::normal_distribution<> dist;

    std::vector<MatrixXd> Y0_init(d);
    if (Y0.empty()) {
        for (int i = 0; i < d; ++i) {
            Y0_init[i] = MatrixXd::NullaryExpr(r[i], n[i] * r[i + 1], [&]() { return dist(rng); });
        }
    } else {
        Y0_init = Y0;
    }

    if (with_rank) {
        return {Y0_init, r};
    } else {
        return {Y0_init, {}};
    }
}