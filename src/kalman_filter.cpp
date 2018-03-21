#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::InitializeFQ(Eigen::MatrixXd &F_in, Eigen::MatrixXd &Q_in) {
  F_ = F_in;
  Q_ = Q_in;
}

void KalmanFilter::InitializeHR(Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in) {
  H_ = H_in;
  R_ = R_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateMatrices(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z, const VectorXd &h) {
  VectorXd y = z - h;
  bool is_sign_negative = std::signbit(y(1));
  const double PI = std::atan(1.0) * 4;
  while (y(1) < -PI || y(1) > PI) {
    if (is_sign_negative == true) {
      y(1) += 2 * PI;
    } else {
      y(1) -= 2 * PI;
    }
  }
  UpdateMatrices(y);
}

void KalmanFilter::UpdateMatrices(const VectorXd &y) {
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
