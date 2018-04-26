
// typedef ArrayRef<int64_t> IntList;

#define AT_CUDA_ENABLED()	1
#define AT_CUDNN_ENABLED()	1
#define AT_MKL_ENABLED()	1

// my_implementation.cpp
#include <torch/torch.h>
#include <unordered_set>
#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Config.h"
// #include "THGeneral.h"
// #include "THC.h"

typedef std::vector<int64_t> MyIntList;

// can use templates as well. But let's keep it
// simple
using scalar_t = float;


// Required for setCuDNNStreamToCurrent
typedef enum
{
    CUDNN_STATUS_SUCCESS                      = 0,
} cudnnStatus_t;

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

#define CUDNN_CHECK(condition)                              \
  do {                                                      \
    cudnnStatus_t status = condition;                       \
    assert(status == CUDNN_STATUS_SUCCESS);		    \
  } while (0)
// cudnnStatus_t CUDNNWINAPI cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t cudnnSetStream     (void* handle, void* streamId);



namespace at { namespace native {
	void raw_cudnn_convolution_forward_out(
					       const at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
					       at::IntList padding, at::IntList stride, at::IntList dilation, int64_t groups,
					       bool benchmark, bool deterministic);
	void cudnn_convolution_add_bias_(CheckedFrom c, const at::TensorArg& output, const at::TensorArg& bias);

	std::vector<int64_t> conv_output_size(
					      at::IntList input_size, at::IntList weight_size,
					      at::IntList padding, at::IntList stride,
					      at::IntList dilation, int64_t groups);

	std::vector<int64_t> conv_input_size(
				     at::IntList output_size, at::IntList weight_size,
				     at::IntList padding, at::IntList output_padding, at::IntList stride, at::IntList dilation,
				     int64_t groups);

	std::vector<int64_t> conv_weight_size(
				      at::IntList input_size, at::IntList output_size,
				      at::IntList padding, IntList output_padding, at::IntList stride, at::IntList dilation,
				      int64_t groups);
	cudnnHandle_t getCudnnHandle();

	at::Tensor narrowGroup(const at::Tensor& t, int dim, int group_idx, int64_t groups);


	// Important: Uncomment!
	typedef void* cudaStream_t;
	typedef void THCState;
	cudaStream_t THCState_getCurrentStream(THCState *state);
	inline void setCuDNNStreamToCurrent() {
	    // TODO: Should getCurrentStream be a method on Context?

	    // IMPORTANT: Uncomment!!

	    // globalcontext: /export/scratch1/hendriks/projects/pytorch/aten/src/ATen/Context.h
	    //
	    CUDNN_CHECK(cudnnSetStream(getCudnnHandle(), THCState_getCurrentStream(at::globalContext().thc_state)));
	};

    }}


at::Tensor unique_float(at::Tensor input_) {
    // only works for floats
    AT_ASSERT(input_.type().scalarType() == at::ScalarType::Float, "input must be a float tensor");
    // and CPU tensors
    AT_ASSERT(!input_.type().is_cuda(), "input must be a CPU tensor");

    // make the input contiguous, to simplify the implementation
    at::Tensor input = input_.contiguous();

    // get the pointer that holds the data
    scalar_t* input_data = input.data<scalar_t>();
    // let's use a function from the std library to implement
    // the unique function
    std::unordered_set<scalar_t> set(input_data, input_data + input.numel());

    // create the output tensor, with size set.size()
    at::Tensor output = input.type().tensor({static_cast<int64_t>(set.size())});
    scalar_t* output_data = output.data<scalar_t>();
    // copy the content of the set to the output tensor
    std::copy(set.begin(), set.end(), output_data);

    return output;
}

struct ConvParams {
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    bool transposed;
    std::vector<int64_t> output_padding;
    int groups;
    bool benchmark;
    bool deterministic;
    bool cudnn_enabled;

    bool is_strided() const;
    bool is_dilated() const;
    bool is_padded() const;
    bool is_output_padding_neg() const;
    bool is_output_padding_big() const;
    bool is_padding_neg() const;
    void view1d_as_2d();
    bool use_cudnn(const at::Tensor& input) const;
    bool use_mkldnn(const at::Tensor& input) const;
    bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

std::ostream& operator<<(std::ostream & out, const ConvParams& params) {
    out << "ConvParams {"
	<< "  stride = " << at::IntList{params.stride}
    << "  padding = " << at::IntList{params.padding}
    << "  dilation = " << at::IntList{params.dilation}
    << "  transposed = " << params.transposed
    << "  output_padding = " << at::IntList{params.output_padding}
    << "  groups = " << params.groups
    << "  benchmark = " << params.benchmark
    << "  deterministic = " << params.deterministic
    << "  cudnn_enabled = " << params.cudnn_enabled
    << "}";
    return out;
}

auto ConvParams::is_strided() const -> bool {
    bool is_strided = false;
    for (int s : stride) {
	is_strided |= (s != 1);
    }
    return is_strided;
}

auto ConvParams::is_dilated() const -> bool {
    bool is_dilated = false;
    for (int d : dilation) {
	is_dilated |= (d != 1);
    }
    return is_dilated;
}

auto ConvParams::is_padded() const -> bool {
    bool is_padded = false;
    for (int p : padding) {
	is_padded |= (p != 0);
    }
    return is_padded;
}

auto ConvParams::is_output_padding_neg() const -> bool {
    bool is_non_neg = false;
    for (int p : output_padding) {
	is_non_neg |= (p < 0);
    }
    return is_non_neg;
}

auto ConvParams::is_output_padding_big() const -> bool {
    bool is_big = false;
    for (size_t i = 0; i < output_padding.size(); i++) {
	is_big |= (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
    }
    return is_big;
}

auto ConvParams::is_padding_neg() const -> bool {
    bool is_non_neg = false;
    for (int p : padding) {
	is_non_neg |= (p < 0);
    }
    return is_non_neg;
}


auto ConvParams::view1d_as_2d() -> void {
    if (stride.size() == 1) {
	stride.insert(stride.begin(), 1);
	padding.insert(padding.begin(), 0);
	dilation.insert(dilation.begin(), 1);
	output_padding.insert(output_padding.begin(), 0);
    }
}

auto ConvParams::use_cudnn(const at::Tensor& input) const -> bool {
#if AT_CUDNN_ENABLED()
    if (!input.type().is_cuda() || !cudnn_enabled) {
	return false;
    }
    if (deterministic && is_dilated()) {
	// cudnn doesn't support deterministic dilated convolution fully yet
	return false;
    }
    if (is_dilated()) {
	// NOTE: extra parenthesis around numbers disable clang warnings about dead code
	// This only works in cudnn version > 6, which was originally checked.
	// We do not really require the check because we assume pytorch
	// was installed using conda, which automatically installs a
	// newer version of cudnn.
	return !is_output_padding_big();
    }
    return !is_output_padding_big();
#else
    (void)input; // avoid unused parameter warning
#endif
    return false;
}

auto ConvParams::use_mkldnn(const at::Tensor& input) const -> bool {
#if AT_MKLDNN_ENABLED()
    return input.type().backend() == kCPU &&
	input.type().scalarType() == kFloat && // only on CPU Float Tensors
	!is_dilated() && // doesn't support dilation
	!transposed && // or transposed tensors
	input.ndimension() == 4 && // must be in NCHW format
	groups == 1;
#endif
    return false;
}

// We currently only have depthwise support for the case where groups ==
// nInputPlane and nInputPlane == nOutputPlane (the latter due to the lack of
// a depthwise multiplier)
auto ConvParams::is_depthwise(
			      const at::Tensor& input, const at::Tensor& weight) const -> bool {
    return input.type().is_cuda() &&
	!transposed &&
	input.ndimension() == 4 &&
	input.size(1) == groups &&
	groups > 1 && // no point if there is only a single group
	weight.size(0) % input.size(1) == 0; // output channels must be a multiple of input channels
}


static void check_input_shape_forward(const at::Tensor& input,
                                      const at::Tensor& weight, const at::Tensor& bias,
                                      int64_t groups, bool transposed) {
    int64_t k = input.ndimension();
    int64_t weight_dim = weight.ndimension();

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

    if (!transposed) {
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
    } else { // transposed
	if (input.size(1) != weight.size(0)) {
	    std::stringstream ss;
	    ss << "Given transposed=" << transposed << ", weight" << weight.sizes()
	       << ", so expected input" << input.sizes() << " to have "
	       << weight.size(0) << " channels, but got " << input.size(1)
	       << " channels instead";
	    throw std::runtime_error(ss.str());
	}
	if (bias.defined() && (bias.ndimension() != 1 || bias.size(0) != weight.size(1) * groups)) {
	    std::stringstream ss;
	    ss << "Given transposed=" << transposed << ", weight of size " << weight.sizes()
	       << ", expected bias to be 1-dimensional with " << weight.size(1) * groups << " elements"
	       << ", but got bias of size " << bias.sizes() << " instead";
	    throw std::runtime_error(ss.str());
	}
    }
}

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
    if (tensor.ndimension() != 3) throw std::runtime_error("expected 3D tensor");
    return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
    if (tensor.ndimension() != 4) throw std::runtime_error("expected 4D tensor");
    return tensor.squeeze(2);
}


static at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
    if (!tensor.defined()) {
	return at::Tensor();
    }
    int64_t n = tensor.sizes()[dim] / groups;
    return tensor.narrow(dim, n * g, n).contiguous();
}

static inline std::vector<int64_t> convolution_expand_param_if_needed(at::IntList list_param, const char *param_name, int64_t expected_dim) {
    if (list_param.size() == 1) {
	return std::vector<int64_t>(expected_dim, list_param[0]);
    } else if ((int64_t) list_param.size() != expected_dim) {
	std::ostringstream ss;
	ss << "expected " << param_name << " to be a single integer value or a "
	   << "list of " << expected_dim << " values to match the convolution "
	   << "dimensions, but got " << param_name << "=" << list_param;
	throw std::runtime_error(ss.str());
    } else {
	return list_param.vec();
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

  auto output_t = input->type().tensor(
				       at::native::conv_output_size(input->sizes(), weight->sizes(),
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
    at::native::setCuDNNStreamToCurrent();
    at::CheckedFrom c = "cudnn_convolution";
    auto output_t = cudnn_convolution_forward(c, input, weight, padding, stride,
					      dilation, groups, benchmark, deterministic);
    if (bias->defined()) {
	at::native::cudnn_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
    }
    return output_t;
}

at::Tensor my_conv(
		   const at::Tensor& input_r, const at::Tensor& weight_r, const at::Tensor& bias_r,
		   MyIntList stride, MyIntList padding, MyIntList dilation,
		   bool transposed_, MyIntList output_padding){

    auto stride_ = at::IntList(stride);
    auto padding_ = at::IntList(padding);
    auto dilation_ = at::IntList(dilation);
    auto output_padding_ = at::IntList(output_padding);

    bool benchmark = false;
    bool deterministic = false;
    bool cudnn_enabled = true;
    int64_t groups_ = 1;

    auto input = input_r.contiguous();
    auto weight = weight_r;
    auto bias = bias_r;
    auto k = input.ndimension();
    int64_t dim = k - 2;

    if (dim <= 0) {
	throw std::runtime_error("input has less dimensions than expected");
    }

    ConvParams params;
    params.stride = convolution_expand_param_if_needed(stride_, "stride", dim);
    params.padding = convolution_expand_param_if_needed(padding_, "padding", dim);
    params.dilation = convolution_expand_param_if_needed(dilation_, "dilation", dim);
    params.transposed = transposed_;
    params.output_padding = convolution_expand_param_if_needed(output_padding_, "output_padding", dim);
    params.groups = groups_;
    params.benchmark = benchmark;
    params.deterministic = deterministic;
    params.cudnn_enabled = cudnn_enabled;

    if (params.is_padding_neg()) throw std::runtime_error("negative padding is not supported");
    if (params.is_output_padding_neg()) throw std::runtime_error("negative output_padding is not supported");

    check_input_shape_forward(input, weight, bias, params.groups, params.transposed);

    if (k == 3) {
	params.view1d_as_2d();
	input = view4d(input);
	weight = view4d(weight);
    }

    auto output = input.type().tensor();

    if (params.is_depthwise(input, weight)) {
	/* output.resize_(output_size(input, weight)); */

	auto kernel_size = weight.sizes().slice(2);
	auto stride = params.stride;
	auto padding = params.padding;
	auto dilation = params.dilation;

	output = at::thnn_conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation);
    } else if (params.use_cudnn(input)) {
#if AT_CUDNN_ENABLED()
	if (input.type() != weight.type()){
	    std::stringstream ss;
	    ss << "Input type (" << input.type().toString() << ") and weight type (" << weight.type().toString() << ") should be the same";
	    throw std::runtime_error(ss.str());
	}
	if (bias.defined() && input.type() != bias.type()){
	    std::stringstream ss;
	    ss << "Input type (" << input.type().toString() << ") and bias type (" << bias.type().toString() << ") should be the same";
	    throw std::runtime_error(ss.str());
	}

	if (params.transposed) {
	    output = at::cudnn_convolution_transpose(
						     input, weight, bias,
						     params.padding, params.output_padding, params.stride, params.dilation, params.groups, params.benchmark, params.deterministic);
	} else {
	    output = /*at*/::cudnn_convolution(input, weight, bias,
					   params.padding, params.stride, params.dilation,
					   params.groups, params.benchmark, params.deterministic);
	}
#endif
    } else if (params.use_mkldnn(input)) {
#if AT_MKLDNN_ENABLED()
	if (input.type() != weight.type()){
	    std::stringstream ss;
	    ss << "Input type (" << input.toString() << ") and weight type (" << weight.toString() << ") should be the same";
	    throw std::runtime_error(ss.str());
	}
	if (bias.defined() && input.type() != bias.type()){
	    std::stringstream ss;
	    ss << "Input type (" << input.toString() << ") and bias type (" << bias.toString() << ") should be the same";
	    throw std::runtime_error(ss.str());
	}

	output = at::mkldnn_convolution(input, weight, bias, params.padding, params.stride, params.dilation);
#endif
    } else {
	if (params.groups == 1) {
	    output = at::_convolution_nogroup(
					      input, weight, bias, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
	} else {
	    std::vector<at::Tensor> outputs(params.groups);
	    for (int g = 0; g < params.groups; ++g) {
		auto input_g = subtensor(input, 1, params.groups, g);
		auto weight_g = subtensor(weight, 0, params.groups, g);
		auto bias_g = subtensor(bias, 0, params.groups, g);
		outputs[g] = at::_convolution_nogroup(
						      input_g, weight_g, bias_g, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
	    }
	    output = at::cat(outputs, 1);
	}
    }

    if (k == 3) {
	output = view3d(output);
    }

    return output;
}



int64_t make_new(MyIntList l, int64_t a, bool return_a) {
    auto l_  = at::IntList(l);
    if (return_a) {
	return a;
    } else {
	return l[a];
    }
}

// this defines the functions exposed to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unique_float", &unique_float, "Unique for float tensors");
    m.def("my_conv", &my_conv, "My own convolution function");
    m.def("make_new", &make_new, "My own make_new function");
}
