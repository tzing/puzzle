#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

#define THRESHOLD_BLACK (2)

bool is_good_point(Mat& img, KeyPoint& kp) {
	assert(img.type() == CV_8UC3);

	const int r = kp.pt.y;
	const int c = kp.pt.x;
	const int radius = kp.size / 2.0 + .5;

	const int i_min = max(r - radius, 0);
	const int i_max = min(r + radius, img.rows);
	const int j_min = max(c - radius, 0);
	const int j_max = min(c + radius, img.cols);

	for (int i = i_min; i < i_max; i++) {
		for (int j = j_min; j < j_max; j++) {
			const auto elem = img.at<Vec3b>(i, j);
			if (elem[0] < THRESHOLD_BLACK || elem[1] < THRESHOLD_BLACK || elem[2] < THRESHOLD_BLACK) {
				return false;
			}
		}
	}

	return true;
}
