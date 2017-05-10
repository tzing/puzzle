#include "utility.hpp"

using namespace std;
using namespace cv;

static const Vec3b BLACK(0, 0, 0);

constexpr int select_min(int base, float cmper) {
	return base < cmper ? base : cmper;
}

constexpr int select_max(int base, float cmper) {
	return base > cmper ? base : (cmper + .5);
}

/*
 * reshape point matrix
 */
void ensurePtShape(InputArray _src, OutputArray _dest) {
	Mat points = _src.getMat();
	if (points.rows == 0) {
		_dest.create(0, 2, CV_32FC1);
		return;
	}

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

	if (points.rows == 0) {
		_points_projected.create(0, 2, CV_32FC1);
		return;
	}

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

	Mat source = _source.getMat();

	// find projection range
	vector<Point2f> cnr_source = {
		Point2f(0,0),
		Point2f(source.cols,0),
		Point2f(source.cols,source.rows),
		Point2f(0,source.rows)
	};

	Mat cnr_on_canvas;
	projectPts(_affine, cnr_source, cnr_on_canvas);

	int r_min = INT_MAX, c_min = INT_MAX;
	int r_max = INT_MIN, c_max = INT_MIN;

	for (int i = 0; i < cnr_on_canvas.rows; i++) {
		r_min = select_min(r_min, cnr_on_canvas.at<float>(i, 1));
		c_min = select_min(c_min, cnr_on_canvas.at<float>(i, 0));
		r_max = select_max(r_max, cnr_on_canvas.at<float>(i, 1));
		c_max = select_max(c_max, cnr_on_canvas.at<float>(i, 0));
	}

	// limit project range
	Mat canvas = _canvas.getMat();
	r_min = min(max(r_min, 0), canvas.rows);
	c_min = min(max(c_min, 0), canvas.cols);
	r_max = max(min(r_max, canvas.rows), 0);
	c_max = max(min(c_max, canvas.cols), 0);

	// find corresponding range on source
	vector<Point2f> pts_on_canvas;
	for (int r = r_min; r < r_max; r++) {
		for (int c = c_min; c < c_max; c++) {
			pts_on_canvas.push_back(Point2f(c, r));
		}
	}

	Mat pts_on_source;
	Mat rev_affine = _affine.getMat().inv();
	projectPts(rev_affine, pts_on_canvas, pts_on_source);

	// project image
	const int length = pts_on_canvas.size();

	#pragma omp parallel for
	for (int i = 0; i < length; i++) {
		// get source index
		const auto r_source = pts_on_source.at<float>(i, 1);
		const int r_up = r_source;
		const int r_bottom = r_up + 1;
		if (r_up < 0 || r_bottom >= source.rows) {
			continue;
		}

		const auto c_source = pts_on_source.at<float>(i, 0);
		const int c_left = c_source;
		const int c_right = c_left + 1;
		if (c_left < 0 || c_right >= source.cols) {
			continue;
		}

		// dump when source is black
		if (source.at<Vec3b>(r_up, c_left) == BLACK) {
			continue;
		}

		// estimate value: bilinear
		const auto difX = c_source - c_left;

		const auto val_UL = source.at<Vec3b>(r_up, c_left);
		const auto val_UR = source.at<Vec3b>(r_up, c_right);
		const auto val_U = val_UL * (1.0 - difX) + val_UR * difX;

		const auto val_BL = source.at<Vec3b>(r_bottom, c_left);
		const auto val_BR = source.at<Vec3b>(r_bottom, c_right);
		const auto val_B = val_BL * (1.0 - difX) + val_BR * difX;

		const auto difY = r_source - r_up;
		const auto val = val_U * (1.0 - difY) + val_B * difY;

		// set value
		const auto pt_target = pts_on_canvas[i];
		const int r_target = pt_target.y;
		const int c_target = pt_target.x;

		canvas.at<Vec3b>(r_target, c_target) = val;
	}

	canvas.copyTo(_canvas);
}
