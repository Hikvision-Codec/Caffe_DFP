#ifndef CAFFE_BASE_RISTRETTO_LAYER_HPP_
#define CAFFE_BASE_RISTRETTO_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"

#include "caffe/proto/caffe.pb.h"
namespace caffe {

/**
 * @brief Provides quantization methods used by other Ristretto layers.
 */
template <typename Dtype>
class BaseRistrettoLayer{
 public:
  explicit BaseRistrettoLayer(const LayerParameter& param);
 protected:
  void QuantizeLayerOutputs_cpu(Dtype* data, const int count);
  void QuantizeLayerOutputs_gpu(Dtype* data, const int count);
  void QuantizeWeights(vector<shared_ptr<Blob<Dtype> > > weights_quantized, const bool bias_term = true);
  /**
   * @brief Trim data to fixed point.
   * @param fl The number of bits in the fractional part.
   */
  void Trim2FixedPoint_simulation_cpu(Dtype* data, const int cnt, const int bit_width, int fl);
  void Trim2FixedPoint_simulation_gpu(Dtype* data, const int cnt, const int bit_width, int fl);
  
  void Trim2FixedPoint_invQ(Dtype* data, const int cnt, const int bit_width, int fl);
  /**
   * @brief Generate random number in [0,1) range.
   */
  // The number of bits used for dynamic fixed point parameters and layer
  // activations.
  int bw_params_, bw_layer_out_;
  // The fractional length of dynamic fixed point numbers.
  int fl_weights_, fl_biases_, fl_layer_out_;
  bool quanti_out_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_RISTRETTO_LAYER_HPP_
