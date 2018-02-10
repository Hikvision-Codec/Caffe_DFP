#ifndef CAFFE_ELTWISE_RISTRETTO_LAYER_HPP_
#define CAFFE_ELTWISE_RISTRETTO_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/base_ristretto_layer.hpp"
namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseRistrettoLayer : public EltwiseLayer<Dtype>, 
      public BaseRistrettoLayer<Dtype> {
 public:
  explicit EltwiseRistrettoLayer(const LayerParameter& param): EltwiseLayer<Dtype>(param),
      BaseRistrettoLayer<Dtype>(param){};
  virtual inline const char* type() const { return "EltwiseRistretto"; }

 protected:
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

#endif  // CAFFE_ELTWISE_LAYER_HPP_
