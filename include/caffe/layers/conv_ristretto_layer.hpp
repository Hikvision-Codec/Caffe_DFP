#ifndef CAFFE_CONV_RISTRETTO_LAYER_HPP_
#define CAFFE_CONV_RISTRETTO_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionRistrettoLayer : public ConvolutionLayer<Dtype>,
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit ConvolutionRistrettoLayer(const LayerParameter& param): 
    ConvolutionLayer<Dtype>(param),BaseRistrettoLayer<Dtype>(param){};
  virtual inline const char* type() const { return "ConvolutionRistretto"; }
  virtual void post_process_params();

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};
}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
