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
