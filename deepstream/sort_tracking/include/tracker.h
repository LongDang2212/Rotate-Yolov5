#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "track.h"
#include "munkres.h"
#include "utils.h"
#include <omp.h>


class Tracker {
public:
    Tracker();
    ~Tracker() = default;

    static float CalculateIou(const cv::Rect& det, const cv::Rect& det1);
    static float CalculateIouRotated(const cv::RotatedRect& det, const Track& track);

    static void HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,
                           size_t nrows, size_t ncols,
                           std::vector<std::vector<float>>& association);

/**
 * Assigns detections to tracked object (both represented as bounding boxes)
 * Returns 2 lists of matches, unmatched_detections
 * @param detection
 * @param tracks
 * @param matched
 * @param unmatched_det
 * @param iou_threshold
 */
    static void AssociateDetectionsToTrackers(const std::vector<BBox>& detection,
                                       std::map<int, Track>& tracks,
                                       std::map<int, BBox>& matched,
                                       std::vector<BBox>& unmatched_det,
                                       float iou_threshold = 0.1);

    void Run(const std::vector<BBox>& detections);

    std::map<int, Track> GetTracks();

private:
    // Hash-map between ID and corresponding tracker
    std::map<int, Track> tracks_;

    // Assigned ID for each bounding box
    int id_;
};

float circle_iou(float a[3], float b[3]);