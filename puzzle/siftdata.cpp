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
}

/*
 * destructor
 */
SiftData::~SiftData() {
	_keypoints.clear();
}

#pragma endregion

void SiftData::align_to(SiftData& target, OutputArray affine) {
	assert(!is_empty);
	assert(!target.is_empty);

	// calc knn
	vector<IdxPair> neighbors;
	knn((*this)._descriptor, target._descriptor, neighbors);

	// run ransac
	ransac(neighbors, (*this)._keypoints, target._keypoints, affine);
}

