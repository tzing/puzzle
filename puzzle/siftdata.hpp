#pragma once

#include <opencv2/core/core.hpp>

/*
 * a class that handles the data and process i need in this project
 */
class SiftData
{
public:
	const bool is_empty = false;

	SiftData();
	SiftData(cv::InputArray image);

	~SiftData();

	void align_to(SiftData& target, cv::OutputArray affine);

private:
	cv::Mat _descriptor;
	std::vector<cv::KeyPoint> _keypoints;
};
