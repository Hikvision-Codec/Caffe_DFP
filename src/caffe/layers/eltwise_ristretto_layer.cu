#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_ristretto_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  EltwiseLayer<Dtype>::Forward_gpu(bottom, top);
  if (this->quanti_out_ == true)
  {
    Dtype* top_data = top[0]->mutable_gpu_data();
    this->QuantizeLayerOutputs_gpu(top_data, top[0]->count());
  }
  
}

template <typename Dtype>
void EltwiseRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //NOT IMPLEMENT
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseRistrettoLayer);

}  // namespace caffe
