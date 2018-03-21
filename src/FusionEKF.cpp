#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initial transition matrix F_, dt is redefined after every new measurement
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;

  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000;

  Q_ = MatrixXd(4, 4);

  // noise in the process covariance matrix Q
  noise_ax = 9;
  noise_ay = 9;

  // measurement noise covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0, 0, 0.0225;

  // measurement noise covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0, 0, 0.0009, 0, 0, 0, 0.09;

  Hj_ = MatrixXd(3, 4);
  h_ = VectorXd(3, 1);

  // measurement laser matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0, 0, 1, 0, 0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // create a 4D state vector, we don't know yet the values of the x state
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates,
      // px = rho * cos(phi), py = rho * sin(phi)
      ekf_.x_ << measurement_pack.raw_measurements_[0] *
                     cos(measurement_pack.raw_measurements_[1]),
          measurement_pack.raw_measurements_[0] *
              sin(measurement_pack.raw_measurements_[1]),
          0, 0;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0],
          measurement_pack.raw_measurements_[1], 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // initial state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) /
             1000000.0;  // dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // modify the F matrix so that the time is integrated
  F_(0, 2) = dt;
  F_(1, 3) = dt;

  // set the process noise covariance matrix Q
  Q_ = MatrixXd(4, 4);
  Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0, 0, dt_4 / 4 * noise_ay,
      0, dt_3 / 2 * noise_ay, dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0, 0,
      dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.InitializeFQ(F_, Q_);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // the sensor data is used to perform the update step for the state and
  // covariance matrices.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.InitializeHR(Hj_, R_radar_);
    h_ = tools.ConvertToPolar(ekf_.x_);

    ekf_.UpdateEKF(measurement_pack.raw_measurements_, h_);

  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.InitializeHR(H_laser_, R_laser_);

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  //  cout << "x_ = " << endl << ekf_.x_ << endl;
  //  cout << "P_ = " << endl << ekf_.P_ << endl;
}
