import cupy as cp

def uniform_rand(lower, higher, dimensions):
    data = cp.random.rand(dimensions[0], dimensions[1]) * abs(higher - lower)
    data += lower
    return data

def xavier_init(lower, higher, shape):
    fan_in, fan_out = shape
    limit = cp.sqrt(6 / (fan_in + fan_out))
    return cp.random.uniform(-limit, limit, size=shape)

def he_init(lower, higher, shape):
    fan_in = shape[0]
    std = cp.sqrt(2 / fan_in)
    return cp.random.normal(0, std, size=shape)