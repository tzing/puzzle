#include "ransac.hpp"
#include <algorithm>
#include <iostream>
#include <ctime>
#include "utility.hpp"

#define RANSAC_ROUND			(1000)				// round of RANSAC

#define NUM_REQ_HOMOGRAPHY		(4)					// required points to build a homography
#define HOMOGRAPHY_H_HEIGHT (NUM_REQ_HOMOGRAPHY<<1)	// height of H when build homography
#define HOMOGRAPHY_H_WIDTH		(9)					// width of H when build homography

#define THRESHOLD_GOODRESULT	(5)					// distance less this threshold would consider as good match result

using namespace std;
using namespace cv;

/*
 * assistnat func for get selected points from selected index array and orignial data array
 */
inline void get_selected_points(vector<IdxPair>& selected_idx, vector<KeyPoint>& kp_base, vector<KeyPoint>& kp_target, OutputArray pt_base_selected, OutputArray pt_target_selected) {
	vector<Point2f> pt_base(selected_idx.size());
	vector<Point2f> pt_target(selected_idx.size());

	for (int i = 0; i < selected_idx.size(); i++) {
		pt_base[i] = kp_base[selected_idx[i].first].pt;
		pt_target[i] = kp_target[selected_idx[i].second].pt;
	}

	Mat(pt_base).reshape(1).copyTo(pt_base_selected);
	Mat(pt_target).reshape(1).copyTo(pt_target_selected);
}

/*
 *	calc homography
 */
void calc_homography(Mat& kp_base, Mat& kp_target, OutputArray affine) {
	assert(kp_base.rows == NUM_REQ_HOMOGRAPHY);
	assert(kp_base.type() == CV_32FC1);
	assert(kp_target.rows == NUM_REQ_HOMOGRAPHY);
	assert(kp_target.type() == CV_32FC1);

	// build martix H
	Mat H(HOMOGRAPHY_H_HEIGHT, HOMOGRAPHY_H_WIDTH, CV_32FC1);
	for (int i = 0; i < NUM_REQ_HOMOGRAPHY; i++) {
		auto x1 = kp_base.at<float>(i, 0);
		auto y1 = kp_base.at<float>(i, 1);
		auto x2 = kp_target.at<float>(i, 0);
		auto y2 = kp_target.at<float>(i, 1);

		float data[] = {
			x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2,
			0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2
		};

		Mat(2, HOMOGRAPHY_H_WIDTH, CV_32FC1, data).copyTo(
			H(Rect(0, i << 1, HOMOGRAPHY_H_WIDTH, 2))
		);
	}

	// get affine matrix
	Mat eig_val, eig_vec;
	eigen(H.t()*H, eig_val, eig_vec);

	eig_vec
		.row(eig_vec.rows - 1)
		.reshape(0, 3)
		.copyTo(affine);
}

void ransac(vector<IdxPair>& _knn_pairs, vector<KeyPoint>& kp_base, vector<KeyPoint>& kp_target, OutputArray best_affine) {
#ifdef _DEBUG
	clog << "[RANSAC_START] ";
	auto tic = clock();
#endif

	assert(_knn_pairs.size() > 0);
	assert(kp_base.size() > 0);
	assert(kp_target.size() > 0);

	// copy vector
	vector<IdxPair> knn_pairs(_knn_pairs);

	int best_score = -1;
	for (int round = 0; round < RANSAC_ROUND; round++) {
		// random pick 4
		random_shuffle(knn_pairs.begin(), knn_pairs.end());

		// calc homography
		Mat pt_base_train;
		Mat pt_tar_train;
		get_selected_points(vector<IdxPair>(knn_pairs.begin(), knn_pairs.begin() + NUM_REQ_HOMOGRAPHY), kp_base, kp_target, pt_base_train, pt_tar_train);

		Mat affine;
		calc_homography(pt_base_train, pt_tar_train, affine);

		// proejct other points
		Mat pt_base_test;
		Mat pt_tar_test;
		get_selected_points(vector<IdxPair>(knn_pairs.begin() + NUM_REQ_HOMOGRAPHY + 1, knn_pairs.end()), kp_base, kp_target, pt_base_test, pt_tar_test);

		Mat pt_projected;
		projectPts(affine, pt_base_test, pt_projected);
		
		// - compare with target
		auto diff = pt_projected - pt_tar_test;

		int score = 0;
		for (int i = 0; i < pt_projected.rows; i++) {
			if (norm(diff.row(i)) < THRESHOLD_GOODRESULT) {
				score++;
			}
		}

		// save best
		if (score > best_score) {
			best_score = score;
			affine.copyTo(best_affine);
		}
	}

#ifdef _DEBUG
	auto toc = clock();
	clog << "[RANSAC_FINISH] " << (float)(toc - tic) / CLOCKS_PER_SEC << "sec elasped" << endl;
#endif
}
