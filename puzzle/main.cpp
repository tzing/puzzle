#include <iostream>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "getopt.hpp"

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
	// parse arguments & input data
	Mat img_sample;
	Mat img_target;
	string filename_output;

	int opt_char;
	while ((opt_char = getopt(argc, argv, "s:t:o:")) != -1) {
		switch (opt_char) {
		// sample image
		case 's':
			clog << "load sample image: " << optarg << endl;
			img_sample = imread(optarg);
			break;

		// target image
		case 't':
			clog << "load target image: " << optarg << endl;
			img_target = imread(optarg);
			break;

		// output filename
		case 'o':
			clog << "set output filename: " << optarg << endl;
			filename_output = optarg;
			break;
		}
	}

	vector<Mat> img_puzzles;
	for (; optind < argc; optind++) {
		clog << "load puzzle image: " << argv[optind] << endl;
		img_puzzles.push_back(imread(argv[optind]));
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

	// output

	clog << "done!" << endl;
	RETURN(0);
}