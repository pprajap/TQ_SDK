#ifndef TTOPT_H
#define TTOPT_H

#include <Eigen/Dense>
#include <vector>
#include <functional>

using namespace Eigen;

std::pair<VectorXi, double> ttopt(
    std::function<std::pair<VectorXd, VectorXd>(const MatrixXi&, const VectorXi&, double, double)> f,
    const std::vector<int>& n, int rank = 4, int evals = -1, 
    const std::vector<MatrixXd>& Y0 = {}, int seed = 42, double fs_opt = 1.0,
    bool add_opt_inner = true, bool add_opt_outer = false, bool add_opt_rect = false,
    bool add_rnd_inner = false, bool add_rnd_outer = false, 
    const std::vector<MatrixXi>& J0 = {}, bool is_max = false);

std::pair<VectorXi, double> ttopt_find(
    const MatrixXi& I, const VectorXd& y, const VectorXd& opt, 
    const VectorXi& i_opt, double y_opt, double opt_opt, bool is_max = false);

MatrixXd ttopt_fs(const MatrixXd& y, double y0 = 0.0, double opt = 1.0);

std::pair<std::vector<MatrixXd>, std::vector<int>> ttopt_init(
    const std::vector<int>& n, int rank, const std::vector<MatrixXd>& Y0 = {}, 
    int seed = 42, bool with_rank = false);

#endif // TTOPT_H