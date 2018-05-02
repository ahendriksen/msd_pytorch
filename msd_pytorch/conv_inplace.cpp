// cudnn_convolutions.cpp
#include <torch/torch.h>
#include <unordered_set>
#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Config.h"
#include <torch/csrc/autograd/variable.h>
#include <cuda.h>
#include <THC/THCGeneral.h>

// TODO: explain why I have this typedef..
typedef std::vector<int64_t> MyIntList;

//
// cudnn.h
//

// Required for setCuDNNStreamToCurrent
typedef enum
{
    CUDNN_STATUS_SUCCESS                      = 0,
} cudnnStatus_t;

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;
#if defined (__cplusplus)
extern "C" {
#endif
// cudnnStatus_t CUDNNWINAPI cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId);
}

// Adapted from caffe2/core/common_cudnn.h
#define CUDNN_CHECK(condition)                              \
  do {                                                      \
    cudnnStatus_t status = condition;                       \
    assert(status == CUDNN_STATUS_SUCCESS);		    \
  } while (0)

// The following definitions are from
// ATen/native/cudnn/Conv.cpp. These definitions are not part of the
// public pytorch api.
namespace at { namespace native {
	void raw_cudnn_convolution_forward_out(const at::Tensor& output,
					       const at::Tensor& input,
					       const at::Tensor& weight,
					       at::IntList padding, at::IntList stride, at::IntList dilation,
					       int64_t groups,
					       bool benchmark, bool deterministic);

	void raw_cudnn_convolution_backward_input_out(const at::Tensor& grad_input,
						      const at::Tensor& grad_output,
						      const at::Tensor& weight,
						      at::IntList padding, at::IntList stride, at::IntList dilation,
						      int64_t groups,
						      bool benchmark, bool deterministic);

	void raw_cudnn_convolution_backward_weight_out(const at::Tensor& grad_weight,
						       const at::Tensor& grad_output,
						       const at::Tensor& input,
						       at::IntList padding, at::IntList stride, at::IntList dilation,
						       int64_t groups,
						       bool benchmark, bool deterministic);

	void cudnn_convolution_add_bias_(CheckedFrom c, const at::TensorArg& output, const at::TensorArg& bias);

	std::vector<int64_t> conv_output_size(at::IntList input_size, at::IntList weight_size,
					      at::IntList padding, at::IntList stride,
					      at::IntList dilation, int64_t groups);
    }}


// From ATen/cudnn/Handles.h.
// Included here because not every system has the cudnn headers.
namespace at { namespace native {
	//
	cudnnHandle_t getCudnnHandle();
    }}

// From ATen/cudnn/utils.h:
// Included here because not every system has the cudnn headers.
inline void setCuDNNStreamToCurrent() {
    // TODO: Should getCurrentStream be a method on Context?
    CUDNN_CHECK(cudnnSetStream(at::native::getCudnnHandle(), THCState_getCurrentStream(at::globalContext().thc_state)));
};

static void check_input_shape_forward(const at::Tensor& input,
                                      const at::Tensor& weight, const at::Tensor& bias){
    int64_t k = input.ndimension();
    int64_t weight_dim = weight.ndimension();
    int64_t groups = 1;

    if (weight_dim != k) {
	std::stringstream ss;
	ss << "Expected " << k << "-dimensional weight for " << k
	   << "-dimensional input " << input.sizes() << ", but got weight of size "
	   << weight.sizes() << " instead";
	throw std::runtime_error(ss.str());
    }
    if (weight.size(0) < groups) {
	std::stringstream ss;
	ss << "Given groups=" << groups << ", expected weight to be at least "
	   << groups << " at dimension 0, but got weight of size " << weight.sizes()
	   << " instead";
	throw std::runtime_error(ss.str());
    }


    if (input.size(1) != (weight.size(1) * groups)) {
	std::stringstream ss;
	ss << "Given groups=" << groups << ", weight" << weight.sizes()
	   << ", so expected input" << input.sizes() << " to have "
	   << (weight.size(1) * groups) << " channels, but got " << input.size(1)
	   << " channels instead";
	throw std::runtime_error(ss.str());
    }
    if (bias.defined() && (bias.ndimension() != 1 || bias.size(0) != weight.size(0))) {
	std::stringstream ss;
	ss << "Given weight of size " << weight.sizes()
	   << ", expected bias to be 1-dimensional with " << weight.size(0) << " elements"
	   << ", but got bias of size " << bias.sizes() << " instead";
	throw std::runtime_error(ss.str());
    }
}


at::Tensor cudnn_convolution_forward(
				     at::CheckedFrom c,
				     const at::TensorArg& input, const at::TensorArg& weight,
				     at::IntList padding, at::IntList stride, at::IntList dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
    at::checkAllSameType(c, {input, weight});
    at::checkAllSameGPU(c, {input, weight});

    auto output_t = input->type().tensor(at::native::conv_output_size(input->sizes(), weight->sizes(),
								    padding, stride, dilation, groups));

  // Avoid ambiguity of "output" when this is being used as backwards
  at::TensorArg output{ output_t, "result", 0 };

  // TODO: Uncomment this and do check!
  // convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  at::Tensor weight_contig = weight->contiguous();

  at::native::raw_cudnn_convolution_forward_out(
      *output, *input, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

at::Tensor cudnn_convolution(const at::Tensor& input_t, const at::Tensor& weight_t, const at::Tensor& bias_t,
			     at::IntList padding, at::IntList stride, at::IntList dilation,
			     int64_t groups, bool benchmark, bool deterministic){
    at::TensorArg input  { input_t,  "input",  1 }, weight { weight_t, "weight", 2 }, bias   { bias_t,   "bias",   3 };
    setCuDNNStreamToCurrent();
    at::CheckedFrom c = "cudnn_convolution";
    auto output_t = cudnn_convolution_forward(c, input, weight, padding, stride,
					      dilation, groups, benchmark, deterministic);
    if (bias->defined()) {
	at::native::cudnn_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
    }
    return output_t;
}


void cudnn_convolution_backward_data_(const at::Tensor& grad_output,
				     const at::Tensor& grad_input,
				     const at::Tensor& weight,
				     MyIntList padding,
				     MyIntList stride,
				     MyIntList dilation){
    auto stride_ = at::IntList(stride);
    auto padding_ = at::IntList(padding);
    auto dilation_ = at::IntList(dilation);
    int64_t groups = 1;
    bool benchmark = false;
    bool deterministic = false;

    setCuDNNStreamToCurrent();

    at::native::raw_cudnn_convolution_backward_input_out(grad_input, grad_output, weight, padding_, stride_, dilation_,
					     groups, benchmark, deterministic);
}

void cudnn_convolution_backward_weight_(const at::Tensor& grad_output,
				       const at::Tensor& input,
				       const at::Tensor& grad_weight,
				       MyIntList padding,
				       MyIntList stride,
				       MyIntList dilation){
    auto stride_ = at::IntList(stride);
    auto padding_ = at::IntList(padding);
    auto dilation_ = at::IntList(dilation);
    int64_t groups = 1;
    bool benchmark = false;
    bool deterministic = false;

    setCuDNNStreamToCurrent();

    at::native::raw_cudnn_convolution_backward_weight_out(grad_weight, grad_output, input, padding_, stride_, dilation_,
					     groups, benchmark, deterministic);
}

at::Tensor cudnn_convolution_backward_bias(const at::Tensor& grad_output) {
    return at::cudnn_convolution_backward_bias(grad_output);
}


void cudnn_convolution_full_forward(const at::Tensor& input_t,
				    const at::Tensor& weight_t,
				    const at::Tensor& bias_t,
				    at::Tensor& output_t,
				    MyIntList padding,
				    MyIntList stride,
				    MyIntList dilation){
    auto stride_ = at::IntList(stride);
    auto padding_ = at::IntList(padding);
    auto dilation_ = at::IntList(dilation);
    int64_t groups = 1;
    at::TensorArg input  { input_t,  "input",  1 };
    at::TensorArg weight { weight_t, "weight", 2 };
    at::TensorArg bias   { bias_t,   "bias",   3 };

    check_input_shape_forward(input_t, weight_t, bias_t);

    setCuDNNStreamToCurrent();
    at::CheckedFrom c = "cudnn_convolution_full_forward";

    at::checkAllSameType(c, {input, weight});
    at::checkAllSameGPU(c, {input, weight});

    // check output_t size
    auto esize = at::native::conv_output_size(input->sizes(), weight->sizes(), padding_, stride_, dilation_, groups);
    at::IntList expected_size = at::IntList(esize);
    at::IntList output_size = output_t.sizes();

    if (! expected_size.equals(output_size)) {
    	std::ostringstream oss;
    	oss << "Expected output tensor to have size " << expected_size
	    << ". Got " << output_size << " instead."
    	    << " (while checking arguments for " << c << ")";
    	throw std::runtime_error(oss.str());
    }

    // Avoid ambiguity of "output" when this is being used as backwards
    at::TensorArg output{ output_t, "result", 0 };

    // See #4500
    at::Tensor weight_contig = weight->contiguous();

    bool benchmark = false;
    bool deterministic = false;

    at::native::raw_cudnn_convolution_forward_out(*output, *input, weight_contig,
    						  padding_, stride_, dilation_, groups,
    						  benchmark, deterministic);

    if (bias->defined()) {
    	at::native::cudnn_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
    }
    // No need to return anything :)
}


// this defines the functions exposed to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cudnn_convolution_full_forward", &cudnn_convolution_full_forward, "cudnn full forward");
    m.def("cudnn_convolution_backward_data_", &cudnn_convolution_backward_data_, "cudnn backward data");
    m.def("cudnn_convolution_backward_weight_", &cudnn_convolution_backward_weight_, "cudnn backward weight");
    m.def("cudnn_convolution_backward_bias", &cudnn_convolution_backward_bias, "cudnn backward bias");
}
