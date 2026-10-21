import numpy as np
try:
    import scipy
    print("scipy", scipy.__version__)
except ImportError:
    print("no scipy")
print("numpy", np.__version__)
