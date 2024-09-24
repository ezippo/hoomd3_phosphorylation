import numpy as np
from tba import *
from tba.util.testing import data

from numpy.testing import assert_array_equal

def test_tba_filter():
    # replace simple transition region
    trj = np.array([1, 1, 2, 2, 3, 3])
    assert_array_equal([1, 1, 1, 3, 3, 3],transition_filter_state_trj(trj, tps_state_num=2, split_state_tp=True)) 
    
    # correctly filter if ended in transition region
    trj = np.array([2, 1, 1, 3, 3])
    assert_array_equal([1, 1, 1, 3, 3],
                       transition_filter_state_trj(trj, tps_state_num=2, split_state_tp=True))

    # check more then 3 states
    trj = np.array([1, 2, 2, 3, 2, 2, 4, 5])
    assert_array_equal([1, 1, 3, 3, 3, 4, 4, 5],
                       transition_filter_state_trj(trj, tps_state_num=2, split_state_tp=True))

    # unequal division of transition path
    trj = np.array([1, 2, 2, 2, 3])
    assert_array_equal([1, 1, 1, 3, 3], transition_filter_state_trj(trj, tps_state_num=2, split_state_tp=True))

    # short transition path
    trj = np.array([1, 2, 3, 3, 3])
    assert_array_equal([1, 1, 3, 3, 3],
                       transition_filter_state_trj(trj, tps_state_num=2, split_state_tp=True))

    # instantenous transition
    trj = np.array([1, 1, 3, 3, 3])
    assert_array_equal(trj, transition_filter_state_trj(trj, tps_state_num=2, split_state_tp=True, return_trans_start_end=False))
