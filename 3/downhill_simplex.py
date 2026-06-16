import numpy as np
from sorter import Sorter

def downhill_simplex(func:callable, x_init:np.ndarray, relerr:float=1e-4):
    N = len(x_init) - 1
    N_inv = 1/N
    x_points = x_init.copy()

    # compute y_values corresponding to the initial x_values
    y_vals = func(x_points)
    # sort the y_init values:
    y_vals, indx = Sorter.quicksort(None, y_vals, make_indx=True)
    # sort the x_values according to their associated y_values
    x_points = x_points[indx] 

    centroid = N_inv*np.sum(x_points[:-1]) # exclude the worst point!

    # check whether target accuracy is reached:
    accuracy = 2 * np.abs(y_vals[0] - y_vals[-1]) / np.abs(y_vals[0] + y_vals[-1])
    if accuracy < relerr:
        return x_points
    
    # propose new point by flipping it wrt centroid
    x_new = 2*centroid - x_points[-1] 
    y_new = func(x_new)

    if y_vals[0] <= y_new < y_vals[1]: # point is decent
        x_points[-1] = x_new
        y_vals[-1] = y_new

    elif y_new < y_vals[0]: # point is best
        x_exp = 2*x_new - centroid # try even further
        y_exp = func(x_exp)
        if y_exp < y_new:
            x_points[-1] = x_exp
            y_vals[-1] = y_exp
        else:
            x_points[-1] = x_new
            y_vals[-1] = y_new

    else:
        x_new = 0.5*(centroid + x_points[-1])
        y_new = func(x_new)
        if y_new < y_vals[-1]:
            x_points[-1] = x_new
            y_vals[-1] = y_new  
                  
    
