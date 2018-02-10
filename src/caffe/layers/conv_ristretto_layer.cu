#include <vector>
#include <cuda_runtime.h>

#include "caffe/util/im2col.hpp"
#include "caffe/layers/conv_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {   
  // Do forward propagation
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    // Trim layer output
    if (this->quanti_out_ == true) { 
      this->QuantizeLayerOutputs_gpu(top_data, top[i]->count());
    }
  }
}

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
 //NOT IMPLEMENT
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionRistrettoLayer);

}  // namespace caffe
