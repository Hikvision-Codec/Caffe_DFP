#ifndef CAFFE_CONCAT_RISTRETTO_LAYER_HPP_
#define CAFFE_CONCAT_RISTRETTO_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/base_ristretto_layer.hpp"

namespace caffe {

/**
 * @brief Takes at least two Blob%s and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatRistrettoLayer : public ConcatLayer<Dtype>, 
       public BaseRistrettoLayer<Dtype> {
 public:
  explicit ConcatRistrettoLayer(const LayerParameter& param): ConcatLayer<Dtype>(param),
      BaseRistrettoLayer<Dtype>(param){};
  virtual inline const char* type() const { return "ConcatRistretto"; }

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

#endif  // CAFFE_CONCAT_LAYER_HPP_
