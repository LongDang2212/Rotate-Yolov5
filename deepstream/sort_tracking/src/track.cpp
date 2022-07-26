#include "track.h"

Track::Track() : kf_(10, 5)
{

    /*** Define constant velocity model ***/
    // state - center_x, center_y, angle, area, aspect ratio,  v_cx, v_cy, v_angle, v_area, v_aspect_ratio
    kf_.F_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    // Give high uncertainty to the unobservable initial velocities
    kf_.P_ << 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 10, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 10, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 10000, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 10000, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 10000, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 10000, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 10000;

    kf_.H_ << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0;

    kf_.Q_ << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.0001, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00001;

    kf_.R_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 10, 0, 0,
        0, 0, 0, 10, 0,
        0, 0, 0, 0, 10;
}

// Get predicted locations from existing trackers
// dt is time elapsed between the current and previous measurements
void Track::Predict()
{
    kf_.Predict();

    // hit streak count will be reset
    if (coast_cycles_ > 0)
    {
        hit_streak_ = 0;
    }
    // accumulate coast cycle count
    coast_cycles_++;
}

// Update matched trackers with assigned detections
void Track::Update(const BBox &bbox)
{

    // get measurement update, reset coast cycle count
    coast_cycles_ = 0;
    // accumulate hit streak count
    hit_streak_++;

    // observation - center_x, center_y, area, ratio
    Eigen::VectorXd observation = ConvertBboxToObservation(bbox);
    kf_.Update(observation);
}

// Create and initialize new trackers for unmatched detections, with initial bounding box
void Track::Init(const BBox &bbox)
{
    this->conf = bbox.conf;
    kf_.x_.head(5) << ConvertBboxToObservation(bbox);
    hit_streak_++;
}

/**
 * Returns the current bounding box estimate
 * @return
 */
BBox Track::GetStateAsBbox() const
{
    return ConvertStateToBbox(kf_.x_);
}

float Track::GetNIS() const
{
    return kf_.NIS_;
}

/**
 * Takes a bounding box in the form [x, y, width, height] and returns z in the form
 * [x, y, w, h, a] where x,y is the centre of the box and s is the scale/area and r is
 * the aspect ratio
 *
 * @param bbox
 * @return
 */
Eigen::VectorXd Track::ConvertBboxToObservation(const BBox &B) const
{
    auto bbox = B.bbox;
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(5);
    auto width = static_cast<float>(bbox.size.width);
    auto height = static_cast<float>(bbox.size.height);
    float center_x = bbox.center.x;
    float center_y = bbox.center.y;
    float rad = bbox.angle * M_PI / 180;
    observation << center_x, center_y, rad, width * height, width / height;
    return observation;
}

/**
 * Takes a bounding box in the centre form [x,y,w,h,a] and returns it in the form
 * of rotated bounding box
 *
 * @param state
 * @return
 */
BBox Track::ConvertStateToBbox(const Eigen::VectorXd &state) const
{
    // state - center_x, center_y, width, height, angle,  v_cx, v_cy, v_width, v_height, v_angle
    auto area = std::max(0, static_cast<int>(state[3]));
    auto aspect_ratio = static_cast<float>(state[4]);
    float width = sqrt(area * aspect_ratio);
    float height = area / width;
    auto x = static_cast<int>(state[0]);
    auto y = static_cast<int>(state[1]);
    float angle = static_cast<float>(state[2]) * 180 / M_PI;
    // std::cout<<"\n"<<angle;
    cv::RotatedRect rect(cv::Point2f(x, y), cv::Size(width, height), angle);
    return BBox{rect, this->conf};
}