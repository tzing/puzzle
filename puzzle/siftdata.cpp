#include "siftdata.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>  // sift
#include <opencv2/highgui/highgui.hpp>		// imshow
#include <iostream>
#include "knn.hpp"
#include "ransac.hpp"
#include "utility.hpp"
#include "log.hpp"

using namespace std;
using namespace cv;

bool is_good_point(Mat& img, KeyPoint& kp);
void blur(InputArray _src, OutputArray _dst);

// feature extractor
static auto gExtractor = xfeatures2d::SiftDescriptorExtractor::create();

/*
 *	constructor, made for `vector<MatWithDescriptor>` usage
 */
SiftData::SiftData() : is_empty(true) {
}

/*
 * constructor
 *
 * input cv::Mat and compute descriptor
 */
SiftData::SiftData(InputArray _source, string name) : image(_source.getMat()) {
	assert(!_source.empty());
	assert(_source.type() == CV_8UC3);

	// log
	log("extract feature from " + name);

	// assign variable
	_name = name;
	_source.copyTo(image);

	// blur
	Mat sift_img;
#ifdef _USE_BLUR
	blur(_source, sift_img);
#else
	sift_img = _source.getMat();
#endif // _USE_BLUR

	// extract feature
	vector<KeyPoint> keypoints;
	Mat descriptor;
	gExtractor->detectAndCompute(sift_img, noArray(), keypoints, descriptor);

	// drop keypoint those located on blank area
	vector<int> pass;

	Mat img = _source.getMat();
	for (int i = 0; i < keypoints.size(); i++) {
		if (is_good_point(img, keypoints[i])) {
			pass.push_back(i);
		}
	}

	// save valid keypoint & descriptor
	_keypoints = vector<KeyPoint>(pass.size());
	_descriptor = Mat(pass.size(), 128, CV_32FC1);
	for (int i = 0; i < pass.size(); i++) {
		const int idx = pass[i];
		_keypoints[i] = keypoints[idx];
		descriptor.row(idx).copyTo(_descriptor.row(i));
	}

#ifdef __ENABLE_KEYPOINT
	// draw keypoint
	Mat cavnas;
	drawKeypoints(_source, _keypoints, cavnas, Scalar(0, 0, 255));

	// show image
	string window_name = name + ": keypoint";
	cv::imshow(window_name, cavnas);

	// wait & destory window
	waitKey();
	destroyWindow(window_name);
#endif // __ENABLE_KEYPOINT
}

/*
 * destructor
 */
SiftData::~SiftData() {
	_name.clear();
	_descriptor.release();
	_keypoints.clear();
}

/*
 * align image
 */
void SiftData::align_to(SiftData& target, OutputArray affine) {
	assert(!is_empty);
	assert(!target.is_empty);
	
	// log
	log("aligning " + _name +" -> " + target._name);

	// calc knn
	logd("[KNN] start");
	vector<IdxPair> neighbors;
	knn((*this)._descriptor, target._descriptor, neighbors);
	logd("[KNN] finish", true);

	// run ransac
	logd("[RANSAC] start");
	vector<IdxPair> select_pairs;
	ransac(neighbors, (*this)._keypoints, target._keypoints, affine, select_pairs);
	logd("[RANSAC] finish", true);
}
