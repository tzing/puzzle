#pragma once

#include <opencv2/core/core.hpp>

typedef std::pair<int, int> KnnIdxPair;

void knn(cv::Mat& desp_base, cv::Mat& desp_target, std::vector<KnnIdxPair>& knn_pairs);
