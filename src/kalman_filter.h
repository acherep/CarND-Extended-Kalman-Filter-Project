#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include <cmath>
#include "Eigen/Dense"

class KalmanFilter {
 public:
  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * InitializeFQ Initializes Kalman filter for the predict step
   * @param F_in Transition matrix
   * @param Q_in Process covariance matrix
   */
  void InitializeFQ(Eigen::MatrixXd &F_in, Eigen::MatrixXd &Q_in);

  /**
   * InitializeHR Initializes Kalman filter for the update step
   * @param H_in Measurement matrix or function
   * @param R_in Measurement covariance matrix
   */
  void InitializeHR(Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   */
  void Predict();

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1,
   * @param h The measurement function at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z, const Eigen::VectorXd &h);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1,
   * @param h The measurement function at k+1
   */
  void UpdateMatrices(const Eigen::VectorXd &y);
};

#endif /* KALMAN_FILTER_H_ */
