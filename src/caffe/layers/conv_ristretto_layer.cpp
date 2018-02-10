#include <vector>

#include "caffe/layers/conv_ristretto_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  this->force_nd_im2col_ = conv_param.force_nd_im2col();
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = this->channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  this->num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (this->reverse_dimensions()) {
    this->conv_out_channels_ = this->channels_;
    this->conv_in_channels_ = this->num_output_;
  } else {
    this->conv_out_channels_ = this->num_output_;
    this->conv_in_channels_ = this->channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = this->conv_out_channels_;
  weight_shape[1] = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_);

  bias_shape.push_back(1);

  
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->kernel_dim_ = this->blobs_[0]->count(1);
  this->weight_offset_ =
      this->conv_out_channels_ * this->kernel_dim_ / this->group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
 
}
template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::post_process_params()
{
    this->QuantizeWeights(this->blobs_, this->bias_term_);
}

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Do forward propagation
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    // Trim layer output
    if (this->quanti_out_ == true) {
      this->QuantizeLayerOutputs_cpu(top_data, top[i]->count());
    }
  }
}

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    //NOT IMPLEMENT
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionRistrettoLayer);
#endif

INSTANTIATE_CLASS(ConvolutionRistrettoLayer);

}  // namespace caffe

