#include "siftdata.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>  // sift
#include <opencv2/highgui/highgui.hpp>		// imshow
#include <iostream>
#include "knn.hpp"
#include "ransac.hpp"
#include "utility.hpp"

using namespace std;
using namespace cv;

// feature extractor
static auto gExtractor = xfeatures2d::SiftDescriptorExtractor::create();

#pragma region constructor / destructor

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

	_name = name;
	_source.copyTo(_image);

	gExtractor->detectAndCompute(_source, noArray(), _keypoints, _descriptor);
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

#pragma endregion

/*
 * show image
 */
void SiftData::imshow(InputArray image, string window_name) {
#ifdef _DEBUG
	cv::imshow(_name+": "+window_name, image);
	clog << "[imshow] " << _name << ": " << window_name << endl;
#endif // _DEBUG
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

#ifdef __ENABLE_KEYPOINT
	// get selected keypoints
	auto getKps = [&](SiftData& _source, int get_idx(IdxPair)) {
		return complexGet<IdxPair, KeyPoint, KeyPoint>(select_pairs, _source._keypoints, get_idx, [](KeyPoint kp) {return kp; });
	};

	auto pt_base_train = getKps(*this, IdxPair_::getBaseIdx);
	auto pt_tar_train = getKps(target, IdxPair_::getTargetIdx);

	// display keys
	auto show = [](SiftData& dat, vector<KeyPoint>& kps) {
		Mat cavnas;
		drawKeypoints(dat._image, dat._keypoints, cavnas, Scalar(0, 0, 255));
		drawKeypoints(cavnas, kps, cavnas, Scalar(0, 255, 0));
		dat.imshow(cavnas, "keypoints");
	};

	show(*this, pt_base_train);
	show(target, pt_tar_train);

	clog << "PAUSED. press to cont." << endl;
	waitKey();
#endif // __ENABLE_KEYPOINT
}
