#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

constexpr int kernel_size = 7;
constexpr float sigma_out = 3;
constexpr float sigma_in = 1;

static Mat kernel;

void initialize() {
	if (kernel.rows == kernel_size) {
		return;
	}

	kernel = Mat(kernel_size, kernel_size, CV_32FC1);

	constexpr int ctr = kernel_size >> 1;
	constexpr float sqr_sig_out = sigma_out * sigma_out * 2;
	constexpr float sqr_sig_in = sigma_in * sigma_in * 2;

	float sum = 0;

	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++) {
			const int dX = i - ctr;
			const int dY = j - ctr;

			const float val_out = exp(-((dX*dX) / sqr_sig_out + (dY*dY) / sqr_sig_out));
			const float val_in = exp(-((dX*dX) / sqr_sig_in + (dY*dY) / sqr_sig_in));
			const float val = val_in - val_out;

			kernel.at<float>(i, j) = val;
			sum += val;
		}
	}

	kernel /= sum;
}

void blur(InputArray _src, OutputArray _dst) {
	initialize();

	filter2D(_src, _dst, -1, kernel);
}

