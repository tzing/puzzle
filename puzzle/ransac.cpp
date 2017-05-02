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
 *	calc homography
 */
void calc_homography(InputArray _kp_base, InputArray _kp_target, OutputArray affine) {
	// get points
	Mat kp_base, kp_target;
	ensurePtShape(_kp_base, kp_base);
	ensurePtShape(_kp_target, kp_target);

	assert(kp_base.rows == NUM_REQ_HOMOGRAPHY);
	assert(kp_base.type() == CV_32FC1);
	assert(kp_target.rows == NUM_REQ_HOMOGRAPHY);
	assert(kp_target.type() == CV_32FC1);

	// build martix H
	Mat H(HOMOGRAPHY_H_HEIGHT, HOMOGRAPHY_H_WIDTH, CV_32FC1);
	for (int i = 0; i < NUM_REQ_HOMOGRAPHY; i++) {
		// get x, y data
		auto x1 = kp_base.at<float>(i, 0);
		auto y1 = kp_base.at<float>(i, 1);
		auto x2 = kp_target.at<float>(i, 0);
		auto y2 = kp_target.at<float>(i, 1);

		// build each row
		float data[] = {
			x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2,
			0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2
		};

		// assign to H
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

/*
 * ransac func
 */
void ransac(vector<IdxPair>& _knn_pairs, vector<KeyPoint>& kp_base, vector<KeyPoint>& kp_target, OutputArray _best_affine, vector<IdxPair>& _selected_pair) {
#ifdef _DEBUG
	clog << "[RANSAC_START] ";
	auto tic = clock();
#endif

	assert(_knn_pairs.size() > 0);
	assert(kp_base.size() > 0);
	assert(kp_target.size() > 0);

	// copy vector
	vector<IdxPair> selected_pairs;
	Mat best_affine;

	#pragma omp parallel for
	for (int round = 0; round < RANSAC_ROUND; round++) {
		// random pick 4
		vector<IdxPair> knn_pairs(_knn_pairs);
		random_shuffle(knn_pairs.begin(), knn_pairs.end());

		// calc homography
		auto getPts = [](vector<IdxPair> _idx_objects, vector<KeyPoint> _source, int get_idx(IdxPair)) {
			return complexGet<IdxPair, KeyPoint, Point2f>(_idx_objects, _source, get_idx, [](KeyPoint kp) {return kp.pt; });
		};

		auto first4Pts = vector<IdxPair>(knn_pairs.begin(), knn_pairs.begin() + NUM_REQ_HOMOGRAPHY);
		auto getFirst4Pts = [&](vector<KeyPoint> _source, int get_idx(IdxPair)) {
			return getPts(first4Pts, _source, get_idx);
		};

		auto pt_base_train = getFirst4Pts(kp_base, IdxPair_::getBaseIdx);
		auto pt_tar_train = getFirst4Pts(kp_target, IdxPair_::getTargetIdx);

		Mat affine;
		calc_homography(pt_base_train, pt_tar_train, affine);

		// proejct other points
		auto remainPts = vector<IdxPair>(knn_pairs.begin() + NUM_REQ_HOMOGRAPHY , knn_pairs.end());
		auto getRemainPts = [&](vector<KeyPoint> _source, int get_idx(IdxPair)) {
			return getPts(remainPts, _source, get_idx);
		};

		auto pt_base_test = getRemainPts(kp_base, IdxPair_::getBaseIdx);

		Mat pt_tar_test;
		ensurePtShape(getRemainPts(kp_target, IdxPair_::getTargetIdx), pt_tar_test);

		Mat pt_projected;
		projectPts(affine, pt_base_test, pt_projected);

		// - compare with target
		auto diff = pt_projected - pt_tar_test;

		vector<IdxPair> pairs;
		for (int i = 0; i < pt_projected.rows; i++) {
			if (norm(diff.row(i)) < THRESHOLD_GOODRESULT) {
				pairs.push_back(remainPts[i]);
			}
		}

		// save best
		#pragma omp critical
		{
			if (pairs.size() > selected_pairs.size()) {
				selected_pairs = pairs;
				best_affine = affine;
			}
		}
	}

	best_affine.copyTo(_best_affine);
	_selected_pair = selected_pairs;

#ifdef _DEBUG
	auto toc = clock();
	clog << "[RANSAC_FINISH] " << (float)(toc - tic) / CLOCKS_PER_SEC << "sec elasped, with score=" << selected_pairs.size() << " in " << _knn_pairs.size() << endl;
#endif
}
