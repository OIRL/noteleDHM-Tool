"""
Title: funs.py

Description: This script contains a collection of utility functions for use in DHM hologram processing. The functions in this script were originally developed in MATLAB by Brian Bogue-Jimenez, with assistance from Prof. Ana Doblas and PhD student Raul Castañeda.

Functions:
- holo_read: reads in and preprocesses a hologram image
- FT: calculates the 2D fast Fourier transform of a hologram
- threshold_FT: removes the DC term and applies thresholding to a hologram
- get_plus1: finds the center coordinate of the +1 diffraction order
- filter_center_plus1: applies a filter to the hologram and plots the filtered hologram after compensating with a reference wave (tilting angle)
- binarize_compensated_plus1: takes a complex field as input and returns a binary image by thresholding the real component of the field.
- get_g_and_h: takes a binary image as input and returns the distance between the center of the largest connected component in the image and the true center of the image. The function also plots the image with a rectangle around the largest connected component and a cross at the true center.
- phi_spherical_C: generates a complex-valued wavefront to compensate for the spherical phase factor considering square +1 term (only C).
- phi_spherical_CxCy: generates a complex-valued wavefront to compensate for the spherical phase factor considering rectangular +1 term (Cy != Cx).

Date: January 6, 2022

Authors: Brian Bogue-Jimenez, Carlos Trujillo, and Ana Doblas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
import skimage.transform
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

from timeit import default_timer as timer

from scipy import optimize
#The pattern search method is implemented using the 'Nelder-Mead' of the scipy library. Although it is not the same, since "Nelder–Mead method aka. the simplex method conceptually resembles PS in its narrowing of the search range for multi-dimensional search spaces but does so by maintaining n + 1 points for n-dimensional search spaces, whereas PS methods computes 2n + 1 points (the central point and 2 points in each dimension).", it is the closest python implementation. The exact same method is the 'direct' method of the same linrary, however, this method is deprecated and it is not recomended. See: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
#The 'pattern search' method is implemented using the closest 'Nelder-Mead' method of the scipy library. See: https://en.wikipedia.org/wiki/Pattern_search_(optimization)#:~:text=Pattern%20search%20(also%20known%20as,are%20not%20continuous%20or%20differentiable.
    
#The 'dual_annealing' method implements the MATLAB's 'simulated annealing'. This function implements the Dual Annealing optimization. This stochastic approach combines the generalization of CSA (Classical Simulated Annealing) and FSA (Fast Simulated Annealing) coupled to a strategy for applying a local search on accepted locations. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing

from scipy.optimize import differential_evolution #Emulates the MATLAB's ga. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
from scipy.optimize import NonlinearConstraint #Emulates MATLAB fmincon's 'interior-point' algorithm. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html#scipy.optimize.NonlinearConstraint

from scipy.misc import derivative #Needed for the 'fmin_ncg' method (MATLAB's 'fminunc' equivalent).
from scipy.optimize import fmin_ncg #MATLAB's 'fminunc' equivalent. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_ncg.html#scipy.optimize.fmin_ncg

from scipy.optimize import fsolve #Emulates MATLAB's 'fsolve'. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html#scipy.optimize.fsolve

from scipy.optimize import basinhopping #Function needed for the Pareto Search:
#The basinhopping algorithm is a global optimization algorithm that combines a local optimizer such as L-BFGS-B with a random sampling method, such as Metropolis-Hastings. It allows to set a 'callback' function that is called after each iteration of the optimization. In this callback function, one can check if the current solution is non-dominated, and if it is, you can add it to your Pareto front.

import pyswarms as ps #The particle swarm optimization library for python.

#The parmoo library allow to use surrogate radial basis functions for optimization as MATLAB's 'surrogate'. See: https://parmoo.readthedocs.io/en/latest/modules/surrogates.html
from parmoo import MOOP
from parmoo.optimizers import LocalGPS
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights

def holo_read(filename, vargin):

    # Read in the image file
    holo = imageio.imread(filename)
    # assing scale factor
    scale = vargin
    
    # Resize the image using the scale factor, adding 1 to the width
    # Use the "nearest neighbor" method for resizing (order=0)
    # Preserve the original pixel values and clip the resulting image to the allowed range
    # Use reflection to handle pixels that fall outside of the image
    holo = skimage.transform.resize(holo, (int(holo.shape[0]*scale), int(holo.shape[1]*scale)), order=1, preserve_range=True, clip=True, mode='reflect')
    
    # Get the dimensions of the resized image
    M, N = holo.shape
        
    # If the width is greater than the height, crop equal amounts from the top and bottom
    if M > N:
        cut = (M - N) // 2
        holo = holo[cut:(M-cut),:N]
    # If the height is greater than the width, crop equal amounts from the left and right
    elif M < N:
        cut = (N - M) // 2
        holo = holo[:M,cut:(N-cut)]
    
    # Get the dimensions of the cropped image
    M, N = holo.shape
    
    # Create a grid of pixel coordinates using numpy's meshgrid function
    m, n = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    
    # Return the image and its dimensions, along with the grid information
    return holo, M, N, m, n

def FT(holo):
    # Shift the zero frequency component of the image to the center
    # Take the 2D fast Fourier transform of the shifted image
    # Shift the zero frequency component back to the top left
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))
    
    # Return the transformed image
    return ft

def threshold_FT(FT_holo, M, N, factor):
    # Calculate the intensity (amplitude) of the transformed image
    I = np.sqrt(np.abs(FT_holo))
    
    # Set a region around the center of the image to zero intensity (DC term removal)
    px = 30
    I[M//2-px:M//2+px, N//2-px:N//2+px] = 0
    
    # Display the resulting image
    #plt.figure(); plt.imshow(I, cmap='gray'); plt.title('DC removed');  plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    mi = np.min(np.min(I)); mx = np.max(np.max(I))
    I = 255*(I - mi)/(mx - mi)
    
    # Create a binary image by thresholding the intensity image
    # Compute the histogram
    hist, bins = np.histogram(I, bins=16)
    # Calculate the threshold using Otsu's method
    threshold = threshold_otsu(bins)
    # Binarize the image using the calculated threshold
    #factor = 0.4 #This value is critical. It depends on the sample features. This value must be carefully selected so the +1 D.O. can be properly detected
    BW = np.where(I > threshold*factor, 1, 0)
    
    # Display the resulting binary image
    #plt.figure(); plt.imshow(BW, cmap='gray'); plt.title('Thresholded image');  plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    # Return the binary image
    return BW
  
def get_plus1(bw):

    cc = label(bw, connectivity=1)
    numPixels = [len(cc[cc == i]) for i in range(1, cc.max()+1)]
    numPixels = np.array(numPixels) # convert the list to a NumPy array
    max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the maximum value
    numPixels[max_index] = 0 # set the value at these indices to 0 (+1 order)
    second_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the second largest value (-1 order)
        
    M, N = bw.shape
    for i in range (M):
        for j in range (N):
            if cc[i,j] != max_index[0] + 1 and cc[i,j] != second_max_index[0] + 1:
                bw[i,j] = 0
                cc[i,j] = 0

    terms = regionprops(cc) # compute region properties of the binary image
    plus1 = terms[0].bbox #; print (plus1) # get the bounding box of the first region
    plus_coor = [(plus1[0] + plus1[2]) / 2, (plus1[1] + plus1[3]) / 2] # calculate the center of the bounding box
    M, N = bw.shape # get the size of the binary image
    dc_coor = [M / 2, N / 2] # calculate the center of the image
    p_and_q = abs(np.subtract(plus_coor, dc_coor)) # calculate the absolute difference between the center of the bounding box and the center of the image
    
    #Only if you wanna paint the rectangle and other stuff
    # Create a figure and plot the data
    fig, ax = plt.subplots(); ax.imshow(bw, cmap='gray'); 
    rect = Rectangle((plus1[1],plus1[0]), np.abs(plus1[0] - plus1[2]), np.abs(plus1[1] - plus1[3]), linewidth=3, edgecolor='r', facecolor='none') # Draw the rectangle
    ax.add_patch(rect); plt.title('+1 Difraction order location'); plt.show() # Show the plot
    
    # values for p,q,m and n (min and kemper paper)
    box_size = terms[0].bbox
    m = np.abs(plus1[0] - plus1[2])/2
    n = np.abs(plus1[1] - plus1[3])/2
    p = p_and_q[0]
    q = p_and_q[1]
    #print(f"P: {p} Q: {q}"); print(f"M: {m} N: {n}")
    
    return plus_coor, m, n, p, q

def filter_center_plus1(FT_holo, plus_coor, m, n, Lambda, X, Y, dx, dy, k):
    # Find the shape of the FT_holo array
    M, N = FT_holo.shape
    # Initialize a filter array of zeros with the same shape as FT_holo
    Filter = np.zeros((M, N))
    # Ignore warnings from NumPy
    np.warnings.filterwarnings("ignore")
    # Set the values in the filter array within the specified range to 1
    Filter[int(plus_coor[0] - n):int(plus_coor[0] + n), int(plus_coor[1] - m):int(plus_coor[1] + m)] = 1
    # Apply the filter to the FT_holo array
    FT_FilteringH = FT_holo * Filter
    # Plot the filtered FT hologram using Matplotlib
    #plt.figure(); plt.imshow(np.log((np.abs(FT_FilteringH)**2)), cmap='gray'); plt.title('FT Hologram filter'); plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    # Calculate the angles ThetaXM and ThetaYM
    ThetaXM = np.arcsin((M/2 - plus_coor[1]) * Lambda / (M * dx))
    ThetaYM = np.arcsin((N/2 - plus_coor[0]) * Lambda / (N * dy))
    # Calculate the reference array
    Reference = np.exp(1j * k * (np.sin(ThetaXM) * X * dx + np.sin(ThetaYM) * Y * dy))
    # Invert the Fourier transform of the filtered hologram
    holo_filtered = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(FT_FilteringH)))
    # Multiply the inverted Fourier transform by the reference array
    holoCompensate = holo_filtered * Reference
    # Return the compensated hologram
    return holoCompensate

def binarize_compensated_plus1(I):
    #lets take the real component of this complex field
    R = np.real(I)
    mi = np.min(np.min(R)); mx = np.max(np.max(R))
    R = 255*(R - mi)/(mx - mi)
    # Calculate the Otsu threshold
    threshold = threshold_otsu(R)
    #print (threshold)
    # Create a binary image by thresholding the intensity image
    #factor = 1.15 #This value is critical. It depends on the sample features. This value must be carefully selected so the +1 D.O. can be properly detected
    factor = 1.15
    BW = np.where(R > threshold*factor, 1, 0)
    # Return the binary image
    return BW

def get_g_and_h(bw):

    cc = label(bw, connectivity=1)
    numPixels = [len(cc[cc == i]) for i in range(1, cc.max()+1)]
    numPixels = np.array(numPixels) # convert the list to a NumPy array
    max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the maximum value
    numPixels[max_index] = 0 # set the value at these indices to 0 
    second_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the second largest value (-1 order)
    numPixels[second_max_index] = 0 # set the value at these indices to 0 
    third_max_index = np.unravel_index(numPixels.argmax(), numPixels.shape) # find the indices of the second largest value (-1 order)
        
    M, N = bw.shape
    for i in range (M):
        for j in range (N):
            if cc[i,j] != max_index[0] + 1 and cc[i,j] != second_max_index[0] + 1 and cc[i,j] != third_max_index[0] + 1:
                bw[i,j] = 0
                cc[i,j] = 0


    terms = regionprops(bw)
    best_term_idx = np.argmin([term.eccentricity for term in terms])
    best_term = terms[best_term_idx]
    block_center = ((best_term.bbox[1] + best_term.bbox[3]) / 2, (best_term.bbox[0] + best_term.bbox[2]) / 2)
    #The bbox attribute of a RegionProperties object (best_term) is a tuple that contains the coordinates of the bounding box for the region (top-left and bottom-right).
    
    M, N = bw.shape
    true_center = (M / 2, N / 2)

    g_and_h = np.abs(np.array(block_center) - np.array(true_center))

    #Only if you wanna paint the rectangle and other stuff
    # Create a figure and plot the data
    fig, ax = plt.subplots(); ax.imshow(bw, cmap='gray'); 
    rect = Rectangle((best_term.bbox[1],best_term.bbox[0]), np.abs(best_term.bbox[3]-best_term.bbox[1]), np.abs(best_term.bbox[2]-best_term.bbox[0]), linewidth=3, edgecolor='r', facecolor='none') # Draw the rectangle
    ax.scatter(block_center[0], block_center[1], color='red', marker='x', s=100); 
    ax.scatter(true_center[0], true_center[1], color='blue', marker='x', s=100);
    ax.add_patch(rect); plt.title('Remaining spherical phase factor location'); plt.show(block = False) # Show the plot

    g, h = g_and_h
    #print(f"G: {g} H: {h}")
    
    #plt.figure(); plt.imshow(cc, cmap='gray'); plt.title('3_terms_cc'); plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    #plt.figure(); plt.imshow(bw, cmap='gray'); plt.title('3_terms_bw'); plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    return g, h

def phi_spherical_C(C, g, h, dx, X, Y, Lambda):
    return (np.pi/(Lambda*C))*((X-(g+1))**2 + (Y-(h+1))**2)*(dx**2)
    
def bin_CF_noTele_BAR_1d(fun, seed_cur, holoCompensate, M, N):
    phi_spherical = fun(seed_cur)
    phase_mask = np.exp((-1j)*phi_spherical)
    corrected_image = holoCompensate * phase_mask
    phase = np.angle(corrected_image)
    phase = phase + np.pi
    ib = np.where(phase > 0.5, 1, 0)
    J = M*N - np.sum(ib)
    return J

def std_CF_noTele_BAR_1d(fun, Cx, holoCompensate, M, N):
    phi_spherical = fun(Cx)
    phase_mask = np.exp((-1j)*phi_spherical)
    corrected_image = holoCompensate * phase_mask
    phase = np.angle(corrected_image)
    phase = phase + np.pi
    J = np.std(phase)
    if Cx == 0:
        J = 0.5
    return J
    
def genetic_algorithm(minfunc, lb, ub):
    start = timer()	#Start to count time
    out = differential_evolution(minfunc, bounds = [(lb, ub)]) #, options=options)
    if out.success:
        print("Optimization successful!")
    else:
        print("Optimization failed.")
    Cy_opt = out.x
    print("Processing time differential_evolution:", timer()-start) #Time for differential_evolution execution
    return Cy_opt

def pattern_search(minfunc, Cy):
    start = timer()	#Start to count time
    out = optimize.minimize(fun = minfunc, x0 = Cy, method='Nelder-Mead')
    if out.success:
        print("Optimization successful!")
    else:
        print("Optimization failed.")
    Cy_opt = out.x
    print("Processing time PS:", timer()-start) #Time for PS execution
    return Cy_opt

def fmincon(minfunc, lb, ub, Cy):
    start = timer()	#Start to count time
    nlc = NonlinearConstraint(fun = minfunc, lb = lb, ub = ub)
    out = optimize.minimize(fun = minfunc, x0 = Cy, method='SLSQP', constraints=nlc) #The 'SLSQP' is equivalent to the default 'interior-point' algorithm that MATLAB's 'fmincon' function uses.
    if out.success:
        print("Optimization successful!")
    else:
        print("Optimization failed.")
    Cy_opt = out.x
    print("Processing time FMC:", timer()-start) #Time for FMC execution
    return Cy_opt

def fminunc(minfunc, Cy):
    start = timer()	#Start to count time
    gradient = lambda t: derivative(minfunc, t) #gradient function
    out = fmin_ncg(minfunc, x0 = Cy, fprime=gradient)
    Cy_opt = out
    print("Processing time FMU:", timer()-start) #Time for FMU execution  
    return Cy_opt

def fsolver(minfunc, Cy):
    start = timer()	#Start to count time
    root = fsolve(minfunc, x0 = Cy)
    Cy_opt = root
    print("Processing time FSO:", timer()-start) #Time for FSO execution
    return Cy_opt

def pareto_search(minfunc, Cy):
    start = timer()	#Start to count time
    # Initialize the Pareto front
    pareto_front = []
    # Define the callback function
    def callback(x, f, accept):
        if not any(y[1] < f for y in pareto_front):
            pareto_front.append((x, f))
            print ('optimized Cy = ', x, ' metric value = ', f)
        else:
            print ('No nondominated solution (metric value): ', f)
    # Perform the basinhopping optimization
    minimizer_kwargs = {"method": "BFGS"}
    out = basinhopping(minfunc, x0 = Cy, callback=callback, minimizer_kwargs=minimizer_kwargs)
    # Find the solution with the minimum objective function value in the Pareto front
    min_obj_val = min(pareto_front, key=lambda x: x[1])[1]
    solution = next(s for s in pareto_front if s[1] == min_obj_val)
    Cy_opt = solution[0]
    print("Processing time PTS:", timer()-start) #Time for PTS execution 
    return Cy_opt
    
def particleswarm(minfunc, lb, ub):
    start = timer()	#Start to count time
    # Perform the optimization
    max_bound = ub * np.ones(1)
    min_bound = lb * np.ones(1)
    bounds = (min_bound, max_bound) #Defining the bounds of the search
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # options for the optimization (default c1 and c2 are the cognitive and social parameters respectively, w is the inertia weight. These parameters affect the exploration/exploitation trade-off of the algorithm)
    optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=1, options=options, bounds=bounds)
    out = optimizer.optimize(minfunc, iters=300)
    Cy_opt = out[1] # out[1] is the position of the best particle, which will be the optimal solution of the optimization.
    print("Processing time PSW:", timer()-start) #Time for PSW execution  
    return Cy_opt
    
def simulannealbnd(minfunc, lb, ub):
    start = timer()	#Start to count time
    # Define the bounds of the optimization problem
    bounds = [(lb, ub)]
    # Use the dual_annealing method to minimize the objective function
    out = optimize.dual_annealing(minfunc, bounds)
    Cy_opt = out.x
    print("Processing time SA:", timer()-start) #Time for SA execution  
    return Cy_opt
    
def brute(minfunc, lb, ub):
    start = timer()	#Start to count time
    lb = -15
    ub = 0
    steps = 500
    step = (ub-lb)/steps
    print ("ub :", ub, "lb: ", lb, "step: ", step)
    val_min = 1e6
    for i in range(1, steps):
        val = minfunc(lb + step*i)
        if val < val_min:
            val_min = val
            Cy_opt = lb + step*i

    print("Processing time BRUTE:", timer()-start) #Time for SA execution  
    return Cy_opt
    
def surrogateopt(minfunc, lb, ub):
    start = timer()	#Start to count time surrogateopt
    my_moop = MOOP(LocalGPS)
    # Add a single continuous design variable in the range [lb, ub]
    my_moop.addDesign({'name': "x1", # optional, name
                   'des_type': "continuous", # optional, type of variable
                   'lb': lb, # required, lower bound
                   'ub': ub, # required, upper bound
                   'tol': 1.0e-8 # optional tolerance
                  })

    def sim_func(x):
        return np.array([minfunc(x["x1"]), 1]) # Define the function for the problem (minfunc)
      
    # Add the simulation to the problem
    my_moop.addSimulation({'name': "MySim", # Optional name for this simulation
                       'm': 2, # This simulation has 1 outputs
                       'sim_func': sim_func, # Our sample sim_func from above
                       'search': LatinHypercube, # Use a LH search
                       'surrogate': GaussRBF, # Use a Gaussian RBF surrogate
                       'hyperparams': {}, # Hyperparams passed to internals
                       'sim_db': { # Optional dict of precomputed points
                                  'search_budget': 10 # Set search budget
                                 },
                      })
                           
    my_moop.addObjective({'name': "f1", 'obj_func': lambda x, s: s["MySim"][0]}) # First objective just returns the first simulation output
    my_moop.addAcquisition({'acquisition': UniformWeights, 'hyperparams': {}}) # Add 1 acquisition function
    my_moop.solve(5) # Solve with 5 iterations of ParMOO algorithm
    results = my_moop.getPF() # Extract the results
    Cy_opt = results[0][0] # Save solution
    print("Processing time SGO:", timer()-start) #Time for SGO execution  
    return Cy_opt

def hybrid_ga_ps(minfunc, lb, ub):
    start = timer()	#Start to count time
    out = differential_evolution(minfunc, bounds = [(lb, ub)]) #, options=options)
    if out.success:
        print("GA Optimization successful!")
    else:
        print("GA Optimization failed.")
    Cy_opt_ga = out.x
    
    out2 = optimize.minimize(fun = minfunc, x0 = Cy_opt_ga, method='Nelder-Mead')
    if out2.success:
        print("PS Optimization successful!")
    else:
        print("PS Optimization failed.")
    Cy_opt = out2.x

    print("Processing time hybrid_ga_ps:", timer()-start) #Time for SA execution  
    return Cy_opt  