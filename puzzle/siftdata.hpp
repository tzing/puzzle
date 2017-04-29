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

	void align_to(SiftData& target, OutputArray affine);

private:
	int _size = 0;

	cv::Mat _image;
	cv::Mat _descriptor;
	std::vector<cv::KeyPoint> _keypoints;
};
