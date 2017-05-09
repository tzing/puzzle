#pragma once

#include <opencv2/core/core.hpp>

/*
 * a class that handles the data and process i need in this project
 */
class SiftData
{
public:
	const bool is_empty = false;
	const cv::Mat image;

	SiftData();
	SiftData(cv::InputArray image, std::string image_name = "noname");

	~SiftData();

	void align_to(SiftData& target, cv::OutputArray affine);

private:
	std::string _name;
	cv::Mat _descriptor;
	std::vector<cv::KeyPoint> _keypoints;
};
