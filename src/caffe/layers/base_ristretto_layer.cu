#include "caffe/layers/base_ristretto_layer.hpp"

#include <math.h>
#include <iostream>
namespace caffe {
template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_gpu(Dtype* data,
      const int count) {
Trim2FixedPoint_simulation_gpu(data, count, bw_layer_out_, fl_layer_out_);
}

template <typename Dtype>
__global__ void Trim2FixedPoint_simulation_kernel(Dtype* data, const int cnt,
      const int bit_width, const int fl, Dtype max_data, Dtype min_data) {
	CUDA_KERNEL_LOOP(index, cnt) {
    // Saturate data

    data[index] = fmax(fmin(data[index], max_data), min_data);
    // Round data
    data[index] /= powf(2, -fl);
    data[index] = rint(data[index]);
    data[index] *= powf(2, -fl);
	}
}


template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_simulation_gpu(Dtype* data, const int cnt,
      const int bit_width, int fl) {
  Dtype max_data = (powf(2, bit_width - 1) - 1) * powf(2, -fl);
  Dtype min_data = -powf(2, bit_width - 1) * powf(2, -fl);
  Trim2FixedPoint_simulation_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bit_width, fl, max_data, min_data);
}

// Explicit instantiations
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_gpu(
    double* top_data, const int top_count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_gpu(
    float* top_data, const int top_count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_simulation_gpu(double* data,
    const int cnt, const int bit_width, int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_simulation_gpu(float* data,
    const int cnt, const int bit_width, int fl);
}  // namespace caffe


