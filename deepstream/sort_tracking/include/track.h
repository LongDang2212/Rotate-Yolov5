#pragma once

#include <opencv2/core.hpp>
#include "kalman_filter.h"

struct BBox{
    cv::RotatedRect bbox;
    float conf;
};

class Track {
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void Init(const BBox& bbox);
    void Predict();
    void Update(const BBox& bbox);
    BBox GetStateAsBbox() const;
    float GetNIS() const;

    int coast_cycles_ = 0, hit_streak_ = 0;
    float conf;
private:
    Eigen::VectorXd ConvertBboxToObservation(const BBox&) const;
    BBox ConvertStateToBbox(const Eigen::VectorXd &state) const;
    
    KalmanFilter kf_;
};
