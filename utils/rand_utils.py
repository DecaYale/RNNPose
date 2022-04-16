import numpy as np  

def truncated_normal(u, sigma, min, max, shape=None):
    """ Generate data following truncated normal distribution

    Args:
        u ([type]): mean
        sigma ([type]): var=sigma^2
        min ([type]): lower bound of the truncating range
        max ([type]): higher bound of the truncating range
        shape ([type], optional): [description]. Defaults to None.
    """

    val = min-1 
    while val<min or val>max: #iterative sampling until the first qualified data emerge
        if shape is not None:
            val = sigma*np.random.randn(shape)+u
        else:
            val = sigma*np.random.randn()+u

    assert val != min-1 

    return val
        
