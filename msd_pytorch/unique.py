import torch
from torch.utils.cpp_extension import load as load_ext
import torch.nn as nn
import torch.nn.functional as F

# pass the source files, they will be compiled on the fly
# and will return a python module

_C = load_ext('my_unique_lib', sources=['msd_pytorch/unique.cpp'])

# now can use the functions implemented in C++
make_new = _C.make_new
unique = _C.unique_float
my_conv = _C.my_conv

a = torch.tensor([1.0, 2.0, 1.0])


input = torch.randn(1, 2, 20, 20)
output = torch.randn(1, 1, 20, 20)
weight = torch.randn(1, 2, 3, 3)
bias = torch.randn(1)
dilation = padding = (2, 2)
stride = (1, 1)
transposed = False
groups = 1
benchmark = False
deterministic = False
cudnn_enabled = True

out1 = my_conv(input, weight, bias, (1,), padding, dilation, transposed, padding)
out2 = F.conv1d(input, weight, bias, stride, padding, dilation, groups)

nn.conv2d(
print(unique(a))
# tensor([ 2.,  1.])
