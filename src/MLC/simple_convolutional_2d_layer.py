import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    print(f"input_matrix={input_matrix}")
    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)),  mode="constant")
    print(f"padded_input={padded_input}")

    input_height_padded, input_width_padded = padded_input.shape
    print(input_height_padded, input_width_padded)
    output_height = (input_height_padded - kernel_height) // stride + 1
    output_width = (input_width_padded - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix

input_matrix = np.array([ [1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.], ])
kernel = np.array([ [1., 2.], [3., -1.], ]) 
padding, stride = 0, 1 
expected = np.array([ [ 16., 21., 26., 31.], [ 41., 46., 51., 56.], [ 66., 71., 76., 81.], [ 91., 96., 101., 106.], ])
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(f"output={output}")


