#include "utility.hpp"

using namespace std;
using namespace cv;

static const Vec3b BLACK(0, 0, 0);

/*
 * reshape point matrix
 */
void ensurePtShape(InputArray _src, OutputArray _dest) {
	Mat points = _src.getMat();

	if (points.rows == 1) {
		points = points.t();
	}

	if (points.channels() == 2) {
		points = points.reshape(1);
	}

	assert(points.cols == 2);
	points.copyTo(_dest);
}

/*
 *	project points using affine transform
 *
 *	input:
 *		- affine [3x3 matrix]
 *		- points [nx2 matrix]
 *
 *  output:
 *		- points_projected [nx2 matrix]
 */
void projectPts(InputArray _affine, InputArray _points, OutputArray _points_projected) {
	assert(_affine.type() == CV_32FC1);
	assert(_affine.rows() == 3);
	assert(_affine.cols() == 3);

	// reshape points
	Mat points;
	ensurePtShape(_points, points);

	assert(points.type() == CV_32FC1);
	assert(points.cols == 2);

	// extend a column with scalar 1
	Mat extended(points.rows, 3, CV_32FC1);
	points.copyTo(extended(Rect(0, 0, 2, extended.rows)));
	extended.col(2).setTo(1);

	// affine transform
	auto projected = _affine.getMat() *extended.t();

	// normalized
	Mat normalized = (
		projected(Rect(0, 0, extended.rows, 2)) / repeat(projected.row(2), 2, 1)
	).t();

	// output
	normalized.copyTo(_points_projected);
}

/*
 *	project whole image to the canvus
 */
void projectImage(InputArray _affine, InputArray _source, InputOutputArray _canvas) {
	assert(_source.type() == CV_8UC3);
	assert(_canvas.type() == CV_8UC3);

	// find non-black part on
	Mat source = _source.getMat();
	vector<Point2f> pts_on_source;

	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < source.cols; j++) {
			if (source.at<Vec3b>(i, j) != BLACK) {
				pts_on_source.push_back(Point2f(j, i));
			}
		}
	}

	// project points
	Mat pts_projected;
	projectPts(_affine, Mat(pts_on_source).reshape(1), pts_projected);

	// copy pixel
	Mat canvas = _canvas.getMat();
	for (int i = 0; i < pts_projected.rows; i++) {
		int r_target = pts_projected.at<float>(i, 1);
		int c_target = pts_projected.at<float>(i, 0);
		if (c_target < 0 || r_target < 0 || r_target >= canvas.rows || c_target >= canvas.cols) {
			continue;
		}

		int r_source = pts_on_source[i].y;
		int c_source = pts_on_source[i].x;

		canvas.at<Vec3b>(r_target, c_target) = source.at<Vec3b>(r_source, c_source);
	}

	canvas.copyTo(_canvas);
}
