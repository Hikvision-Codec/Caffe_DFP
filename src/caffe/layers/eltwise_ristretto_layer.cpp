#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_ristretto_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void EltwiseRistrettoLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  EltwiseLayer<Dtype>::Forward_cpu(bottom, top);
  if (this->quanti_out_ == true)
  {
    Dtype* top_data = top[0]->mutable_cpu_data();
    this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
  }
}

template <typename Dtype>
void EltwiseRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  EltwiseLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseRistrettoLayer);
#endif

INSTANTIATE_CLASS(EltwiseRistrettoLayer);
REGISTER_LAYER_CLASS(EltwiseRistretto);
}  // namespace caffe
