#include "knn.hpp"

constexpr int NUM_KNN = 1;		// num of k

using namespace std;
using namespace cv;

typedef pair<int, double> IdxDistantPair;

IdxPair_::IdxPair_() :
	this_idx(-1),
	idx_base(-1),
	idx_target(-1) {
}

IdxPair_::IdxPair_(int _this_idx, int _idx_base, int _idx_target)	:
	this_idx(_this_idx),
	idx_base(_idx_base),
	idx_target(_idx_target) {
}

int IdxPair_::getBaseIdx(IdxPair p) {
	return p->idx_base;
}

int IdxPair_::getTargetIdx(IdxPair p) {
	return p->idx_target;
}

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
void knn(Mat& desp_base, Mat& desp_target, vector<IdxPair>& _knn_pairs) {
	const size_t length = desp_base.rows * NUM_KNN;
	_knn_pairs = vector<IdxPair>(length);

	// iter over each descriptor in base iamge
	int idx_in_knn_pair = 0;

	for (int i = 0; i < desp_base.rows; i++) {
		vector<IdxDistantPair> distants(desp_target.rows);
		const auto row_base = desp_base.row(i);

		// compare with each descriptor in target iamge
		for (int j = 0; j < desp_target.rows; j++) {
			auto row_target = desp_target.row(j);

			// calc distanct: SSE
			auto val = norm(row_base - row_target, NORM_L2);

			// save distance
			distants[j] = make_pair(j, val);
		}

		// sort distants
		sort(distants.begin(), distants.end(), cmp_pair);

		// take k smallest
		for (int j = 0; j < NUM_KNN; j++) {
			_knn_pairs[idx_in_knn_pair] = new IdxPair_(idx_in_knn_pair, i, distants[j].first);
			idx_in_knn_pair++;
		}
	}
}
