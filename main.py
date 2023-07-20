import numpy as np

import hps_phosphorylation.hoomd_util as hu

if __name__=='__main__':
    tmp = hu.compute_center(np.array([[1,2,3],[4,5,6]]))
    print(tmp)
