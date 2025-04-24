import numpy as np

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	v = np.array(a)
    v = np.transpose(v)
    return v.tolist()

