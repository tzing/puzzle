#include "knn.hpp"
#include <iostream>
#include <ctime>

#define NUM_KNN (3)		// knn中要取多少個

using namespace std;
using namespace cv;

typedef pair<int, double> IdxDistantPair;

/*
 *	comparing func for distant data
 */
bool cmp_pair(IdxDistantPair a, IdxDistantPair b)
{
	return a.second < b.second;
}

/*
 *	K Nearest Neighbor algorithm
 */
void knn(Mat& desp_base, Mat& desp_target, vector<KnnIdxPair>& knn_pairs) {
#ifdef _DEBUG
	clog << "KNN_START";
	auto tic = clock();
#endif

	knn_pairs = vector<KnnIdxPair>(desp_base.rows * NUM_KNN);

	// iter over each descriptor in base iamge
	for (int i = 0, idx_in_knn_pair = 0; i < desp_base.rows; i++, idx_in_knn_pair += NUM_KNN) {
		vector<IdxDistantPair> distants(desp_target.rows);
		const auto row_base = desp_base.row(i);

		// compare with each descriptor in target iamge
		for (int j = 0; j < desp_target.rows; j++) {
			auto row_target = desp_target.row(j);

			// calc distanct: SSE
			Mat sq_diff;
			pow(row_base - row_target, 2.0, sq_diff);

			auto val = sum(sq_diff).val[0];

			// save distance
			distants[j] = make_pair(j, val);
		}

		// sort distants
		sort(distants.begin(), distants.end(), cmp_pair);

		// take k smallest
		for (int j = 0; j < NUM_KNN; j++) {
			knn_pairs[idx_in_knn_pair + j] = make_pair(i, distants[j].first);
		}
	}

#ifdef _DEBUG
	auto toc = clock();
	clog << "KNN_FINISH: " << (float)(toc - tic) / CLOCKS_PER_SEC << "sec elasped" << endl;
#endif
}