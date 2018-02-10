#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include "caffe/layers/base_ristretto_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
BaseRistrettoLayer<Dtype>::BaseRistrettoLayer(const LayerParameter& param) {
  this->bw_layer_out_ = param.quantization_param().bw_layer_out();
  this->bw_params_ = param.quantization_param().bw_params();
  this->fl_layer_out_ = param.quantization_param().fl_layer_out();
  this->fl_weights_ = param.quantization_param().fl_weights();
  this->fl_biases_ = param.quantization_param().fl_biases();
  this->quanti_out_ = param.quantization_param().quanti_out();
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights(
      vector<shared_ptr<Blob<Dtype> > > weights_quantized, const bool bias_term) {
  Dtype* weight = weights_quantized[0]->mutable_cpu_data();
  const int cnt_weight = weights_quantized[0]->count();
  Trim2FixedPoint_invQ(weight, cnt_weight, bw_params_, fl_weights_);
  if (bias_term) {
      Trim2FixedPoint_invQ(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), bw_params_, fl_biases_);
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* data, const int count) {
  Trim2FixedPoint_simulation_cpu(data, count, bw_layer_out_,  fl_layer_out_);
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_simulation_cpu(Dtype* data, const int cnt,
      const int bit_width, int fl) {

  Dtype max_data = (powf(2, bit_width - 1) - 1) * powf(2, -fl);
  Dtype min_data = -powf(2, bit_width - 1) * powf(2, -fl);
  Dtype step_forward = powf(2, fl);
  Dtype back_forward = powf(2, -fl);

  Dtype tmp_data = 0;
  for (int index = 0; index < cnt; ++index) {

    tmp_data = data[index];
    tmp_data = std::max(std::min(tmp_data, max_data), min_data);
    tmp_data *= step_forward;
    tmp_data = rint(tmp_data);
    data[index] = tmp_data * back_forward;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_invQ(Dtype* data, const int cnt,
      const int bit_width, int fl) {
  for (int index = 0; index < cnt; ++index) {
    data[index] *= powf(2, -fl);
	}
}
template BaseRistrettoLayer<double>::BaseRistrettoLayer(const LayerParameter& param);
template BaseRistrettoLayer<float>::BaseRistrettoLayer(const LayerParameter& param);
template void BaseRistrettoLayer<double>::QuantizeWeights(
    vector<shared_ptr<Blob<double> > > weights_quantized, const bool bias_term);
template void BaseRistrettoLayer<float>::QuantizeWeights(
    vector<shared_ptr<Blob<float> > > weights_quantized, const bool bias_term);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_cpu(double* data,
    const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_cpu(float* data,
    const int count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_simulation_cpu(double* data,
    const int cnt, const int bit_width, int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_simulation_cpu(float* data,
    const int cnt, const int bit_width, int fl);
}  // namespace caffe
