#include "siftdata.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <utility>
#include <algorithm>
#include "knn.hpp"
#include "ransac.hpp"

using namespace std;
using namespace cv;

// feature extractor
auto gExtractor = xfeatures2d::SiftDescriptorExtractor::create();

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
SiftData::SiftData(InputArray image) {
	assert(!image.empty());

	gExtractor->detectAndCompute(image, noArray(), _keypoints, _descriptor);

	_image = image.getMat();
	_size = _keypoints.size();
}

/*
 * destructor
 */
SiftData::~SiftData() {
	_keypoints.clear();
}

#pragma endregion

void SiftData::align_to(SiftData& base) {
	assert(!is_empty);
	assert(!base.is_empty);

	// calc knn
	vector<IdxPair> neighbors;
	knn(base._descriptor, (*this)._descriptor, neighbors);

	// run ransac
	ransac(neighbors, base._keypoints, (*this)._keypoints, _affine);
}


