#include <iostream>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "getopt.hpp"
#include "siftdata.hpp"
#include "utility.hpp"

#ifdef _DEBUG
#define RETURN(x)	system("pause"); return (x); // it helps on debuging
#else
#define RETURN(x)	return (x);
#endif

using namespace std;
using namespace cv;

/*
 *	main program
 *
 *	usage:
 */
int main(const int argc, char *const argv[]) {
#pragma region parse arguments & read data
	Mat img_sample;
	Mat img_target;
	string filename_output;

	int opt_char;
	while ((opt_char = getopt(argc, argv, "s:t:o:")) != -1) {
		switch (opt_char) {
		case 's': // sample image
			clog << "load sample image: " << optarg << endl;
			img_sample = imread(optarg);
			break;

		case 't': // target image
			clog << "load target image: " << optarg << endl;
			img_target = imread(optarg);
			break;

		case 'o': // output filename
			clog << "set output filename: " << optarg << endl;
			filename_output = optarg;
			break;
		}
	}

	vector<Mat> img_puzzles;
	for (; optind < argc; optind++) {
		clog << "load puzzle image: " << argv[optind] << endl;

		auto img = imread(argv[optind]);
		if (img.empty()) {
			continue;
		}

		img_puzzles.push_back(img);
	}

	// anit-foolish
	if (img_sample.empty()) {
		cerr << "sample image not assigned." << endl;
		RETURN(1);
	}

	if (img_target.empty()) {
		cerr << "target image not assigned." << endl;
		RETURN(1);
	}

	if (img_puzzles.size() == 0) {
		cerr << "no puzzle image assigned." << endl;
		RETURN(2);
	}

#pragma endregion

#pragma region feature extraction
	SiftData dat_sample(img_sample, "sample");
	SiftData dat_target(img_target, "target");

	vector<SiftData> dat_puzzles;
	int c = 1;
	for (auto& img : img_puzzles) {
		char title[10];
		sprintf_s(title, "puzzle%d", c++);
		dat_puzzles.push_back(SiftData(img, title));
	}

#pragma endregion

#pragma region alignment
	Mat affine_target2sample;
	dat_target.align_to(dat_sample, affine_target2sample);
	auto aff_sample_to_target = affine_target2sample.inv();

	vector<Mat> aff_puzzle_to_target;

	for (auto& dat : dat_puzzles) {
		// get affine matrix
		Mat affine;
		dat.align_to(dat_sample, affine);

		// save affine matrix
		aff_puzzle_to_target.push_back(aff_sample_to_target *affine);
	}

#ifdef  __ENABLE_KEYPOINT
	waitKey();
#endif //  __ENABLE_KEYPOINT

#pragma endregion

#pragma region puzzle!
	Mat result(img_target);

	for (int i = 0; i < img_puzzles.size(); i++) {
		projectImage(aff_puzzle_to_target[i], img_puzzles[i], result);
	}

#pragma endregion

	// output
	imshow("result", result);
	clog << "done! press any key to continue" << endl;
	waitKey();
}
