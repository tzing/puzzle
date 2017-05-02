#pragma once

#include <opencv2/core/core.hpp>
#include "knn.hpp"

void ransac(std::vector<IdxPair>& knn_pairs, std::vector<cv::KeyPoint>& kp_base, std::vector<cv::KeyPoint>& kp_target, cv::OutputArray affine, std::vector<IdxPair>& selected_pair);
