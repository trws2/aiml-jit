# Principles of Large Model Quantization Technology: Summary

This article is generated from [this one](https://mp.weixin.qq.com/s/E2itzyivEY-dg0O-7sICnw).

In recent years, with the introduction of Transformer and MOE architectures, deep learning models have easily 
surpassed trillions of parameters, leading to increasingly large models. Therefore, we need some large model 
compression techniques to reduce deployment costs and improve inference performance. Model compression can be 
mainly divided into the following categories: model pruning, knowledge distillation, and model quantization. 
This series will discuss some common large model quantization schemes (such as GPTQ, LLM.int8(), SmoothQuant, AWQ, etc.).

## Basic Principles
### Summary
Model quantization is a technique used to reduce the size and computation of neural network models by converting model parameters (such as weights) from high-precision data types (such as float32) to low-precision data types (such as int8 or fp4). By representing data with fewer bits, model quantization can decrease the model size, thereby reducing memory consumption during inference. Additionally, it can increase inference speed on some processors that perform low-precision operations more quickly, while still maintaining the model's performance.

### Model quantization granularity:

- **Per-tensor (also known as per-layer) quantization**: Each layer or tensor has only one scaling factor, and all values within the tensor are quantized using this scaling factor.

- **Per-channel quantization**: Each channel of the convolution kernel has a different scaling factor.

- **Per-token quantization**: Quantization is performed for each row concerning activations. In large language models (LLMs), this is usually used in conjunction with per-channel quantization, such as: token-wise quantization of activations and channel-wise quantization of weights.

- **Per-group/group-wise**: Quantization is done in groups. As mentioned in "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT," a special case of group quantization treats each dense matrix as a group, where each matrix can have its own quantization range. A more general case is to segment each dense matrix by output neurons, with each contiguous N output neurons forming a group. For example, GPTQ and AWQ use a group size of 128 elements for quantization. Some places also refer to this as sub-channel-wise quantization, which means dividing channels into smaller subgroups to achieve finer granularity in precision control.
