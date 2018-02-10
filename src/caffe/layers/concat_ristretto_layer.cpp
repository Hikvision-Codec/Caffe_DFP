#include <vector>

#include "caffe/layers/concat_ristretto_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/////////////////////
template <typename Dtype>
void ConcatRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ConcatLayer<Dtype>::Forward_cpu(bottom, top);
  if (this->quanti_out_ == true)
  {
    Dtype* top_data = top[0]->mutable_cpu_data();
    this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
  }
}

template <typename Dtype>
void ConcatRistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   ConcatLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(ConcatRistrettoLayer);
#endif

INSTANTIATE_CLASS(ConcatRistrettoLayer);
REGISTER_LAYER_CLASS(ConcatRistretto);

}  // namespace caffe
