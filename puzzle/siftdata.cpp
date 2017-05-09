#include "siftdata.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>  // sift
#include <opencv2/highgui/highgui.hpp>		// imshow
#include <iostream>
#include "knn.hpp"
#include "ransac.hpp"
#include "utility.hpp"

using namespace std;
using namespace cv;

bool is_good_point(Mat& img, KeyPoint& kp);

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
SiftData::SiftData(InputArray _source, string name) {
	assert(!_source.empty());
	assert(_source.type() == CV_8UC3);

	_name = name;
	_source.copyTo(_image);

	vector<KeyPoint> keypoints;
	Mat descriptor;
	gExtractor->detectAndCompute(_source, noArray(), keypoints, descriptor);

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
	clog << "PAUSED. press to cont." << endl;
	waitKey();
	destroyWindow(window_name);
#endif // __ENABLE_KEYPOINT
}

/*
 * destructor
 */
SiftData::~SiftData() {
	_name.clear();
	_image.release();
	_descriptor.release();
	_keypoints.clear();
}

/*
 * align image
 */
void SiftData::align_to(SiftData& target, OutputArray affine) {
	assert(!is_empty);
	assert(!target.is_empty);

	// calc knn
	vector<IdxPair> neighbors;
	knn((*this)._descriptor, target._descriptor, neighbors);

	// run ransac
	vector<IdxPair> select_pairs;
	ransac(neighbors, (*this)._keypoints, target._keypoints, affine, select_pairs);
}
