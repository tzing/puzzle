#pragma once

#include <opencv2/core/core.hpp>

void projectPts(cv::InputArray _affine, cv::InputArray _points, cv::OutputArray _points_projected);
void projectImage(cv::InputArray _affine, cv::InputArray _source, cv::InputOutputArray _canvas);
