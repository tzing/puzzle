#pragma once

#include <opencv2/core/core.hpp>

typedef std::pair<int, int> IdxPair;

void knn(cv::Mat& desp_base, cv::Mat& desp_target, std::vector<IdxPair>& knn_pairs);
