import os
import torch
import random
import numpy as np

def set_seed(seed, deterministic):
        
    #根据seed设置训练结果的随机性
    """
        影响可复现的因素主要有这几个：
        1.随机种子 python，numpy,torch随机种子；环境变量随机种子；GPU随机种子
        2.训练使用不确定的算法
        2.1 CUDA卷积优化——CUDA convolution benchmarking
        2.2 Pytorch使用不确定算法——Avoiding nondeterministic algorithms
    """
    # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
    torch.manual_seed(seed)

    # Set python seed
    random.seed(seed)

    # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
    # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
    # need to be seeded consistently as well.)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        
        # torch.use_deterministic_algorithms(True)

        # If you wish to continue to enable determinism, but allow certain operations to be nondeterministic, you can use the following setting:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
        # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.
