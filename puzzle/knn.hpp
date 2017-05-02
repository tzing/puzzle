#pragma once

#include <opencv2/core/core.hpp>

struct IdxPair_;
typedef IdxPair_* IdxPair;

struct IdxPair_ {
	const int this_idx;
	const int idx_target;
	const int idx_base;

	IdxPair_();
	IdxPair_(int, int, int);

	static int getBaseIdx(IdxPair);
	static int getTargetIdx(IdxPair);
};

void knn(cv::Mat& desp_base, cv::Mat& desp_target, std::vector<IdxPair>& knn_pairs);
