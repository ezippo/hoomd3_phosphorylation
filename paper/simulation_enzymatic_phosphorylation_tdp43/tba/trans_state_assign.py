import pyprind 
import numpy as np



def cba_1d_single_state_mask(rc, lw, up):
    return (rc >= lw)  & (rc <= up)


def assign_cba_first_frame_1d(rc, state_centers):
    # +1 for 1 based state indeces
    return np.argmin(np.absolute(rc - state_centers)) +1


def cba_multi_states(rc, state_lw_up):
    n_states = state_lw_up.shape[0]
    cba_ar = np.zeros(rc.shape)
    for st_i in pyprind.prog_bar(range(0, n_states)):
        #upper = state_lw_up[st_i+1][0]
        ma = cba_1d_single_state_mask(rc, state_lw_up[st_i][0], state_lw_up[st_i][1])
        cba_ar[ma] = st_i +1
    return cba_ar
    
    
def cba_traj_1d(rc, state_centers, state_width):
    
    state_lw_up = np.column_stack((state_centers - state_width, state_centers + state_width))
    #state_lw_up = np.column_stack((state_centers, state_centers + state_width))
    cba_ar = cba_multi_states(rc, state_lw_up)
    if cba_ar[0] == 0:
        cba_ar[0] = assign_cba_first_frame_1d(rc[0], state_centers)
    return cba_ar


# TBA

def transition_filter_state_trj(cba_ar, split_state_tp=True,
                                return_trans_start_end=False, tps_state_num=0, verbose=False):
    """
    
    Simple and general transition based state assignment
    
    This can return time information as well
    for detailed investigation of transition statistics as needed in the analysis of REMD.
    """
    #c = 0.0
    #tba_ar = np.zeros(cba_ar.shape)
    tba_ar = cba_ar.copy()
    N = len(cba_ar)
    
    not_tps = tba_ar[tba_ar != tps_state_num]
    # catch if we start or end on a TPS
    if tba_ar[-1] == tps_state_num:
       tba_ar[-1] = not_tps[-1]
    
    if tba_ar[0] == tps_state_num:
       tba_ar[0] = not_tps[0]
    
    tps_idx = zeros_stretch(tba_ar, tps_state_num=tps_state_num)
    for tp in tps_idx:
        tba_ar[tp[0]:tp[1]] = tba_ar[tp[0]-1]
    
    if split_state_tp:
       tba_ar = split_tp_ar(tba_ar, tps_idx, verbose)
        
    if return_trans_start_end:
        return tba_ar, tps_idx
    else:
         return tba_ar
            

def split_tp_ar(ar, tps_idx,verbose=False):
    N = len(ar)
    for tp in tps_idx:
        # exclude TPs that don't end with the trajectory
        #if tp[1] < N-1:
        tp_start, tp_end = tp
        tp_len = tp_end - tp_start
        # first half will be longer for odd-lengths arrays
        second_half = tp_len / 2  + tp_len % 2 + tp_start
        if verbose:
#           print('assigned second half of transition path starting at {} to product state {}).format(int(second_half), int(tp_end)))
           print('start of second half of transition path, end transition path')
           print(int(second_half), int(tp_end))
        # assigns second half to product state
        #print tp_end
        ar[int(second_half):int(tp_end)] = ar[int(tp_end)]
    return ar        
    
    
def zeros_stretch(a, tps_state_num=0):
    """
    Parameters:
    """
    # https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    # 0 where a is not zero (stable state), 1 where a is 0 (transition path)
    tps = np.concatenate(([tps_state_num], np.equal(a, tps_state_num).view(np.int8),
                          [tps_state_num]))
    # crucial step is to calculate the discrete differences between neighbours in the array
    abs_neighbour_diff = np.abs(np.diff(tps))
    # return the start and end points of the zero stretches, at these points difference is 1 
    return np.where(abs_neighbour_diff ==1 )[0].reshape(-1,2)

# --------------------------------------------------------------------------------------------    

# convert trajectory in bin indices sequence
def traj_in_binindices(traj, xin=-3, yin=-3, xfin=3, yfin=3, nbins_side=10):
    ''' dividing plane in bins and converting trajectory position in bin indices in lexicographic order:
        x, y rescaled in range [0,nbins_side) and transformed in integers
        bin index = (y / nbins_side) + (x % nbins_side) 
    '''
    nn = len(traj)
    binned_trajx = np.floor((traj[:,0]-xin)*nbins_side/(xfin-xin)).astype(np.int32)
    binned_trajy = np.floor((traj[:,1]-yin)*nbins_side/(yfin-yin)).astype(np.int32) 
    binindices_traj = np.array([ binned_trajx[i] + nbins_side*binned_trajy[i] for i in range(nn) ])
    return binindices_traj

    
def test_binindices_traj(binindices_traj, traj, xin=-3, yin=-3, xfin=3, yfin=3, nbins_side=10, i_start=0, i_end=10, verbose=False):
    ''' testing correspondence between trajectory position and bin index '''
    xstep = (xfin-xin)/nbins_side
    ystep = (yfin-yin)/nbins_side
    test_arr = np.zeros(i_end - i_start +1)
    for i in range(i_start, i_end+1):
        x_min = xin + xstep*(binindices_traj[i]%nbins_side)
        y_min = yin + ystep*(int(binindices_traj[i]/nbins_side))
        x_bool = (x_min <= traj[i][0] < x_min+xstep)
        if not x_bool:
            np.testing.assert_almost_equal(x_min, traj[i][0],err_msg=f"\n bin index: {binindices_traj[i]};  x value: {traj[i][0]}")
        y_bool = (y_min <= traj[i][1] < y_min+ystep)
        if not y_bool:
            np.testing.assert_almost_equal(y_min, traj[i][1],err_msg=f"\n bin index: {binindices_traj[i]};  y value: {traj[i][1]}")
        test_arr[i] = x_bool and y_bool
    sumtest = (-test_arr+1).sum()
    print("TEST BINNED TRAJECTORY SUCCESSFUL! ")
    if verbose==True:
        print(f"Number of failed positions:  {sumtest} ")
        print("Real trajectory:")
        print(traj[i_start:i_end+1])
        print("Bin indices:")
        print(binindices_traj[i_start:i_end+1])
        print("Test array:")
        print(test_arr)
    return sumtest

# inserting tps state for positions too far from cluster center
def inserting_tps_state_2D(dtrajs, traj, cluster_centers, r, tps_state_num=None):
    ''' '''
    tps_dtrajs = np.copy(dtrajs)
    if tps_state_num==None:
        tps_state_num = len(cluster_centers)
    nn = len(traj)
    for i in range(nn):
        cl_index = dtrajs[i]
        dist2 = ((traj[i] - cluster_centers[cl_index])**2).sum()
        if dist2 > r[cl_index]**2:
            tps_dtrajs[i] = tps_state_num
    return tps_dtrajs
        