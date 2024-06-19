#!/usr/bin/env python
# coding: utf-8

# [![在线运行](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_modelarts.png)](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svcjIuMC90dXRvcmlhbHMvemhfY24vYmVnaW5uZXIvbWluZHNwb3JlX3RlbnNvci5pcHluYg==&imageid=b8671c1e-c439-4ae2-b9c6-69b46db134ae)&emsp;[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r2.0/tutorials/zh_cn/beginner/mindspore_tensor.ipynb)&emsp;[![下载样例代码](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_download_code.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r2.0/tutorials/zh_cn/beginner/mindspore_tensor.py)
# &emsp;[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/tutorials/source_zh_cn/beginner/tensor.ipynb)
# 
# [基本介绍](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/introduction.html) || [快速入门](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/quick_start.html) || **张量 Tensor** || [数据集 Dataset](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/dataset.html) || [数据变换 Transforms](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/transforms.html) || [网络构建](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/model.html) || [函数式自动微分](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/autograd.html) || [模型训练](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/train.html) || [保存与加载](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/save_load.html)

# # 张量 Tensor
# 
# 张量（Tensor）是一个可用来表示在一些矢量、标量和其他张量之间的线性关系的多线性函数，这些线性关系的基本例子有内积、外积、线性映射以及笛卡儿积。其坐标在 $n$ 维空间内，有  $n^{r}$ 个分量的一种量，其中每个分量都是坐标的函数，而在坐标变换时，这些分量也依照某些规则作线性变换。$r$ 称为该张量的秩或阶（与矩阵的秩和阶均无关系）。
# 
# 张量是一种特殊的数据结构，与数组和矩阵非常相似。张量（[Tensor](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.Tensor.html)）是MindSpore网络运算中的基本数据结构，本教程主要介绍张量和稀疏张量的属性及用法。

# In[1]:


import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor


# ## 创建张量
# 
# 张量的创建方式有多种，构造张量时，支持传入`Tensor`、`float`、`int`、`bool`、`tuple`、`list`和`numpy.ndarray`类型。
# 
# - **根据数据直接生成**
# 
#     可以根据数据创建张量，数据类型可以设置或者通过框架自动推断。

# In[2]:


data = [1, 0, 1, 0]
x_data = Tensor(data)
print(x_data)

# - **从NumPy数组生成**
# 
#     可以从NumPy数组创建张量。

# In[3]:


np_array = np.array(data)
x_np = Tensor(np_array)


# - **使用init初始化器构造张量**
# 
#     当使用`init`初始化器对张量进行初始化时，支持传入的参数有`init`、`shape`、`dtype`。
# 
#     - `init`: 支持传入[initializer](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore.common.initializer.html)的子类。
# 
#     - `shape`: 支持传入 `list`、`tuple`、 `int`。
# 
#     - `dtype`: 支持传入[mindspore.dtype](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.dtype.html#mindspore.dtype)。

# In[4]:


from mindspore.common.initializer import One, Normal

# Initialize a tensor with ones
tensor1 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=One())
# Initialize a tensor from normal distribution
tensor2 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=Normal())

print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)


#     `init`主要用于并行模式下的延后初始化，在正常情况下不建议使用init对参数进行初始化。
# 
# - **继承另一个张量的属性，形成新的张量**

# In[5]:


from mindspore import ops

x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")


# ## 张量的属性
# 
# 张量的属性包括形状、数据类型、转置张量、单个元素大小、占用字节数量、维数、元素个数和每一维步长。
# 
# - 形状（shape）：`Tensor`的shape，是一个tuple。
# 
# - 数据类型（dtype）：`Tensor`的dtype，是MindSpore的一个数据类型。
# 
# - 单个元素大小（itemsize）： `Tensor`中每一个元素占用字节数，是一个整数。
# 
# - 占用字节数量（nbytes）： `Tensor`占用的总字节数，是一个整数。
# 
# - 维数（ndim）： `Tensor`的秩，也就是len(tensor.shape)，是一个整数。
# 
# - 元素个数（size）： `Tensor`中所有元素的个数，是一个整数。
# 
# - 每一维步长（strides）： `Tensor`每一维所需要的字节数，是一个tuple。

# In[6]:


x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)

print("x_shape:", x.shape)
print("x_dtype:", x.dtype)
print("x_itemsize:", x.itemsize)
print("x_nbytes:", x.nbytes)
print("x_ndim:", x.ndim)
print("x_size:", x.size)
print("x_strides:", x.strides)




# In[7]:
# ## 张量索引
# 
# Tensor索引与Numpy索引类似，索引从0开始编制，负索引表示按倒序编制，冒号`:`和 `...`用于对数据进行切片。

tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))

print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))



# In[8]:
# ## 张量运算
# 
# 张量之间有很多运算，包括算术、线性代数、矩阵处理（转置、标引、切片）、采样等，张量运算和NumPy的使用方式类似，下面介绍其中几种操作。
# 
# > 普通算术运算有：加（+）、减（-）、乘（\*）、除（/）、取模（%）、整除（//）。


x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)




# In[9]:
# `Concat`将给定维度上的一系列张量连接起来。

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

print(output)
print("shape:\n", output.shape)




# In[10]:
# `Stack`则是从另一个维度上将两个张量合并起来。

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])

print(output)
print("shape:\n", output.shape)




# In[11]:
# ## Tensor与NumPy转换
# 
# Tensor可以和NumPy进行互相转换。
# 
# ### Tensor转换为NumPy
# 
# 与张量创建相同，使用 `asnumpy()` 将Tensor变量转换为NumPy变量。

t = ops.ones(5, mindspore.float32)
print(f"t: {t}")
n = t.asnumpy()
print(f"n: {n}")


# ### NumPy转换为Tensor
# 
# 使用`Tensor()`将NumPy变量转换为Tensor变量。

# In[12]:


n = np.ones(5)
t = Tensor.from_numpy(n)


# In[13]:


np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

#######################################################################################
# 选做
# ## 稀疏张量
# 
# 稀疏张量是一种特殊张量，其中绝大部分元素的值为零。
# 
# 在某些应用场景中（比如推荐系统、分子动力学、图神经网络等），数据的特征是稀疏的，若使用普通张量表征这些数据会引入大量不必要的计算、存储和通讯开销。这时就可以使用稀疏张量来表征这些数据。
# 
# MindSpore现在已经支持最常用的`CSR`和`COO`两种稀疏数据格式。
# 
# 常用稀疏张量的表达形式是`<indices:Tensor, values:Tensor, shape:Tensor>`。其中，`indices`表示非零下标元素， `values`表示非零元素的值，shape表示的是被压缩的稀疏张量的形状。在这个结构下，我们定义了三种稀疏张量结构：`CSRTensor`、`COOTensor`和`RowTensor`。

# ### CSRTensor
# 
# `CSR`（Compressed Sparse Row）稀疏张量格式有着高效的存储与计算的优势。其中，非零元素的值存储在`values`中，非零元素的位置存储在`indptr`（行）和`indices`（列）中。各参数含义如下：
# 
# - `indptr`: 一维整数张量, 表示稀疏数据每一行的非零元素在`values`中的起始位置和终止位置, 索引数据类型支持int16、int32、int64。
# 
# - `indices`: 一维整数张量，表示稀疏张量非零元素在列中的位置, 与`values`长度相等，索引数据类型支持int16、int32、int64。
# 
# - `values`: 一维张量，表示`CSRTensor`相对应的非零元素的值，与`indices`长度相等。
# 
# - `shape`: 表示被压缩的稀疏张量的形状，数据类型为`Tuple`，目前仅支持二维`CSRTensor`。
# 
# > `CSRTensor`的详细文档，请参考[mindspore.CSRTensor](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.CSRTensor.html)。
# 
# 下面给出一些CSRTensor的使用示例：

# In[14]:


indptr = Tensor([0, 1, 2])
indices = Tensor([0, 1])
values = Tensor([1, 2], dtype=mindspore.float32)
shape = (2, 4)

# Make a CSRTensor
csr_tensor = CSRTensor(indptr, indices, values, shape)

print(csr_tensor.astype(mindspore.float64).dtype)


# 上述代码会生成如下所示的`CSRTensor`:
# 
# $$
#  \left[
#  \begin{matrix}
#    1 & 0 & 0 & 0 \\
#    0 & 2 & 0 & 0
#   \end{matrix}
#   \right]
# $$

# ### COOTensor
# 
# `COO`（Coordinate Format）稀疏张量格式用来表示某一张量在给定索引上非零元素的集合，若非零元素的个数为`N`，被压缩的张量的维数为`ndims`。各参数含义如下：
# 
# - `indices`: 二维整数张量，每行代表非零元素下标。形状：`[N, ndims]`， 索引数据类型支持int16、int32、int64。
# 
# - `values`: 一维张量，表示相对应的非零元素的值。形状：`[N]`。
# 
# - `shape`: 表示被压缩的稀疏张量的形状，目前仅支持二维`COOTensor`。
# 
# > `COOTensor`的详细文档，请参考[mindspore.COOTensor](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.COOTensor.html)。
# 
# 下面给出一些COOTensor的使用示例：

# In[15]:


indices = Tensor([[0, 1], [1, 2]], dtype=mindspore.int32)
values = Tensor([1, 2], dtype=mindspore.float32)
shape = (3, 4)

# Make a COOTensor
coo_tensor = COOTensor(indices, values, shape)

print(coo_tensor.values)
print(coo_tensor.indices)
print(coo_tensor.shape)
print(coo_tensor.astype(mindspore.float64).dtype)  # COOTensor to float64


# 上述代码会生成如下所示的`COOTensor`:
# 
# $$
#  \left[
#  \begin{matrix}
#    0 & 1 & 0 & 0 \\
#    0 & 0 & 2 & 0 \\
#    0 & 0 & 0 & 0
#   \end{matrix}
#   \right]
# $$
# 
