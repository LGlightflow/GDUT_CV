# coding: utf-8

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul() # 计算两个数组的乘积
        
        #mindspore.Parameter(default_input, name, requires_grad=True, layerwise_parallel=False)
        #Parameter是变量张量，代表在训练网络时，需要被更新的参数。
        #inited_param：返回保存了实际数据的Parameter，如果Parameter原本保存的是MetaTensor，会将其转换为Tensor。
        #name：实例化Parameter时，为其指定的名字。
        #sliced：用在自动并行场景下，表示Parameter里保存的数据是否是分片数据。
        #如果是，就不再对其进行切分，如果不是，需要根据网络并行策略确认是否对其进行切分。
        #is_init：Parameter的初始化状态。在GE后端，Parameter需要一个init graph来从主机同步数据到设备侧，该标志表示数据是否已同步到设备。 此标志仅在GE后端起作用，其他后端将被设置为False。
        #layerwise_parallel：Parameter是否支持layerwise并行。如果支持，参数就不会进行广播和梯度聚合，反之则需要。
        #requires_grad：是否需要计算参数梯度。如果参数需要被训练，则需要计算参数梯度，否则不需要。
        #data： Parameter本身。
        self.z = Parameter(Tensor(np.array([1.0, 1.0, 1.0], np.float32)), name='z') 

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out
    

model = Net()

for m in model.parameters_and_names():
    print(m)

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
result = model(x, y)
print(result)