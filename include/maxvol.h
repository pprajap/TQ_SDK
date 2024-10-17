#ifndef MAXVOL_H
#define MAXVOL_H

#include <Eigen/Dense>

using namespace Eigen;

std::pair<VectorXi, MatrixXd> maxvol(const MatrixXd& A, double e = 1.05, int k = 100);
std::pair<VectorXi, MatrixXd> maxvol_rect(const MatrixXd& A, double e = 1.1, int dr_min = 0, int dr_max = -1, double e0 = 1.05, int k0 = 10);

#endif // MAXVOL_H