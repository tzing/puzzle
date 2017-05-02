#pragma once

#include <opencv2/core/core.hpp>

void ensurePtShape(cv::InputArray _src, cv::OutputArray _dest);

void projectPts(cv::InputArray _affine, cv::InputArray _points, cv::OutputArray _points_projected);
void projectImage(cv::InputArray _affine, cv::InputArray _source, cv::InputOutputArray _canvas);

/*
 * get selected objects by indexs
 */
template<typename _T>
inline std::vector<_T> getByIdx(std::vector<_T> _source, std::vector<int> _idxs) {
	std::vector<_T> sub_vecs(_idxs.size());
	for (int i = 0; i < _idxs.size();i++) {
		auto target_idx = _idxs[i];
		sub_vecs[i] = _source[target_idx];
	}
	return sub_vecs;
}

/*
* get object that refer from index inside other objects
*/
template<typename _TIdx, typename _TSrc, typename _TOut>
inline std::vector<_TOut> complexGet(std::vector<_TIdx> _idx, std::vector<_TSrc> _src, int get_idx(_TIdx), _TOut get_output(_TSrc)) {
	vector<_TOut> output(_idx.size());
	for (int i = 0; i < _idx.size(); i++) {
		auto obj_with_idx = _idx[i];
		auto target_idx = get_idx(obj_with_idx);
		auto target_obj = _src[target_idx];
		output[i] = get_output(target_obj);
	}
	return output;
}
