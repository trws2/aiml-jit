# Principles of Large Model Quantization Technology: Summary

This article is generated from [this one](https://mp.weixin.qq.com/s/E2itzyivEY-dg0O-7sICnw).

In recent years, with the introduction of Transformer and MOE architectures, deep learning models have easily 
surpassed trillions of parameters, leading to increasingly large models. Therefore, we need some large model 
compression techniques to reduce deployment costs and improve inference performance. Model compression can be 
mainly divided into the following categories: model pruning, knowledge distillation, and model quantization. 
This series will discuss some common large model quantization schemes (such as GPTQ, LLM.int8(), SmoothQuant, AWQ, etc.).

# Basic Principles
## Summary
Model quantization is a technique used to reduce the size and computation of neural network models by converting model parameters (such as weights) from high-precision data types (such as float32) to low-precision data types (such as int8 or fp4). By representing data with fewer bits, model quantization can decrease the model size, thereby reducing memory consumption during inference. Additionally, it can increase inference speed on some processors that perform low-precision operations more quickly, while still maintaining the model's performance.

