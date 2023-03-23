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
- bin_CF_noTele_BAR_1d: generates the output value of the cost function based on binarizing the resulting phase image.
- std_CF_noTele_BAR_1d: generates the output value of the cost function based on computing the standard deviation of the resulting phase image.
- fmincon: finds the minimum of a constrained multivariable function. 
- fminunc: finds the minimum scalar value of a non-linear unconstrained multivariable objective function.
- fsolver: returns a vector that minimizes the objective function by solving for the function F(x) = 0.
- simulannealbnd: simulated annealing probabilistic technique well suited for finding the global minimum of a large and discrete search space.
- pareto_search: finds the points in a Pareto front that minimizes two cost functions of a two-dimensional variable. In this case, we use the J1 and J2 cost functions.
- genetic_algorithm: minimizes a cost function given the number of variables in the function by iteratively picking the best population values within the range specified by the bounds.
- pattern_search: algorithm that does not utilize gradients, allowing for the convergence of cost functions that are not continuous or differentiable. 
- hybrid_ga_ps: finds the minimum value of a cost function using first the GA search, and then finetunes with the PS search.
- brute: finds the minimum value of a cost function in the whole search range with a fixed number of steps.

Date: January 6, 2023
Updated: March 10, 2023

Authors: Brian Bogue-Jimenez, Carlos Trujillo, and Ana Doblas
"""

#All operations are perfomed using numpy
import numpy as np
import math

#Ploting and visualizing results
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#IO image files handling
import imageio

#OpenCV library. Image manipulation and visualization.
import cv2

#Image processing operations and functions from SciKit learn library
import skimage.transform
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

#Time counting library
from timeit import default_timer as timer

#The pattern search method is implemented using the 'Nelder-Mead' of the scipy library. Although it is not the same, since "Nelder–Mead method aka. the simplex method conceptually resembles PS in its narrowing of the search range for multi-dimensional search spaces but does so by maintaining n + 1 points for n-dimensional search spaces, whereas PS methods computes 2n + 1 points (the central point and 2 points in each dimension).", it is the closest python implementation. The exact same method is the 'direct' method of the same linrary, however, this method is deprecated and it is not recomended. See: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
#The 'pattern search' method is implemented using the closest 'Nelder-Mead' method of the scipy library. See: https://en.wikipedia.org/wiki/Pattern_search_(optimization)#:~:text=Pattern%20search%20(also%20known%20as,are%20not%20continuous%20or%20differentiable.
#The 'dual_annealing' method implements the MATLAB's 'simulated annealing'. This function implements the Dual Annealing optimization. This stochastic approach combines the generalization of CSA (Classical Simulated Annealing) and FSA (Fast Simulated Annealing) coupled to a strategy for applying a local search on accepted locations. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing
from scipy import optimize
   
#Emulates the MATLAB's ga. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
from scipy.optimize import differential_evolution

#Emulates MATLAB fmincon's 'interior-point' algorithm. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html#scipy.optimize.NonlinearConstraint
from scipy.optimize import NonlinearConstraint 

#Needed for the 'fmin_ncg' method (MATLAB's 'fminunc' equivalent).
from scipy.misc import derivative 

#MATLAB's 'fminunc' equivalent. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_ncg.html#scipy.optimize.fmin_ncg
from scipy.optimize import fmin_ncg

#Emulates MATLAB's 'fsolve'. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html#scipy.optimize.fsolve
from scipy.optimize import fsolve 

#Function needed for the Pareto Search:
#The basinhopping algorithm is a global optimization algorithm that combines a local optimizer such as L-BFGS-B with a random sampling method, such as Metropolis-Hastings. It allows to set a 'callback' function that is called after each iteration of the optimization. In this callback function, one can check if the current solution is non-dominated, and if it is, you can add it to your Pareto front.
from scipy.optimize import basinhopping 

# Define global variables for storing mouse click coordinates
mouse_x = 1
mouse_y = 1

def holo_read(filename, vargin):

    '''
    Variables:
    filename: string variable with the name of the hologram to be process
    vargin: float variable determining a factor [0-1] to scale the input image
    '''
    
    # Read in the image file
    holo = imageio.imread(filename)
    
    '''
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

    '''
    
    # Get the dimensions of the cropped image
    M, N = holo.shape
    
      
    # Create a grid of pixel coordinates using numpy's meshgrid function
    X, Y = np.meshgrid(np.arange(-N//2, N//2), np.arange(-M//2, M//2))
    
    # Return the image and its dimensions, along with the grid information
    return holo, M, N, X, Y

def FT(holo):

    '''
    Variables:
    holo: numpy array with hologram data to be processed.
    '''
    
    # Shift the zero frequency component of the image to the center
    # Take the 2D fast Fourier transform of the shifted image
    # Shift the zero frequency component back to the top left
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))
    
    # Return the transformed image
    return ft

def threshold_FT(FT_holo, M, N):

    '''
    Variables:
    FT_holo: numpy array with hologram spectrum data to be processed
    M and N: integer varaibles representing the number of pixels in each dimension of 'FT_holo'
    '''
    
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
    BW = np.where(I > threshold, 1, 0)
    
    # Display the resulting binary image
    #plt.figure(); plt.imshow(BW, cmap='gray'); plt.title('Thresholded image');  plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    # Return the binary image
    return BW
  
def get_plus1(bw):

    '''
    Variables:
    bw: numpy array with thresholded hologram spectrum data to be processed
    '''
    
    #Isolating the +1 difraction orders
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
    rect = Rectangle((plus1[1],plus1[0]), np.abs(plus1[1] - plus1[3]), np.abs(plus1[0] - plus1[2]), linewidth=3, edgecolor='r', facecolor='none') # Draw the rectangle
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

    '''
    Variables:
    FT_holo: numpy array with hologram spectrum data to be processed
    plus_coor: list with two floating-point values representing the x-coordinate and y-coordinate of the center point of the +1 term
    M and N: integer varaibles representing the number of pixels in each dimension of 'FT_holo'
    Lambda: float variable indicating the illumination wavelength
    X and Y: are two-dimensional numpy arrays of integers with shapes (N, M), where N and M are the dimensions of the mesh.
    dx and dy: float varaibles indicating the pixel pitches along the x- and y-directions.
    k: float variable indicating the wavenumber.
    '''
    
    # Find the shape of the FT_holo array
    M, N = FT_holo.shape
    # Initialize a filter array of zeros with the same shape as FT_holo
    Filter = np.zeros((M, N))

    # Set the values in the filter array within the specified range to 1
    Filter[int(plus_coor[0] - m):int(plus_coor[0] + m), int(plus_coor[1] - n):int(plus_coor[1] + n)] = 1
    # Apply the filter to the FT_holo array
    FT_FilteringH = FT_holo * Filter
    # Plot the filtered FT hologram using Matplotlib
    #plt.figure(); plt.imshow(np.log((np.abs(FT_FilteringH)**2)), cmap='gray'); plt.title('FT Hologram filter'); plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    # Calculate the angles ThetaXM and ThetaYM
    ThetaXM = np.arcsin((M/2 - plus_coor[0]) * Lambda / (M * dx))
    ThetaYM = np.arcsin((N/2 - plus_coor[1]) * Lambda / (N * dy))
    
    # Calculate the reference array
    Reference = np.exp(1j * k * (np.sin(ThetaXM) * X * dx + np.sin(ThetaYM) * Y * dy))
    # Invert the Fourier transform of the filtered hologram
    holo_filtered = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(FT_FilteringH)))
    # Multiply the inverted Fourier transform by the reference array
    holoCompensate = holo_filtered * Reference
    # Return the compensated hologram
    return holoCompensate
    
def binarize_compensated_plus1(I):

    '''
    Variables:
    I: numpy array with hologram spectrum data to be binarized
    '''
    #lets take the real component of this complex field
    R = np.real(I)
    mi = np.min(np.min(R)); mx = np.max(np.max(R))
    R = 255*(R - mi)/(mx - mi)
    # Calculate the Otsu threshold
    threshold = threshold_otsu(R)
    #print (threshold)
    # Create a binary image by thresholding the intensity image
    BW = np.where(R > threshold, 1, 0)
    # Return the binary image
    return BW

def get_g_and_h(bw):

    '''
    Variables:
    bw: numpy array with binarized hologram spectrum data to be processed
    '''
    
    #Segment the spherical phase factor coordinates
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
    true_center = (N / 2, M / 2)

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
    '''
    Variables:
    C: float variable indicatinng the phase factor curvature
    g and h: flaot variables indicating the coordinated of the spherical phase factor
    dx: float variable indicating the pixel pitc in the x direction.
    X and Y: are two-dimensional numpy arrays of integers with shapes (N, M), where N and M are the dimensions of the mesh.
    Lambda: float variable indicating the illumination wavelength
    '''
    return (np.pi/(Lambda*C))*((X-(g+1))**2 + (Y-(h+1))**2)*(dx**2)
    
def bin_CF_noTele_BAR_1d(fun, seed_cur, holoCompensate, M, N, sign):
    '''
    Variables:
    fun: function to optimize
    seed_cur: flaot variable indicating the seed of the search 
    holoCompensate: hologram data to compensate
    M and N: integer varaibles representing the number of pixels in each dimension of 'holoCompensate'
    '''
    phi_spherical = fun(seed_cur)  # Compute phi_spherical by calling the function fun with input seed_cur.
    
    if (sign):
        phase_mask = np.exp((1j)*phi_spherical)  # Compute the phase mask by taking the exponential of 1j times phi_spherical.
    else:
        phase_mask = np.exp((-1j)*phi_spherical)  # Compute the phase mask by taking the exponential of -1j times phi_spherical.
        
    corrected_image = holoCompensate * phase_mask  # Compensate the image by multiplying the hologram with the phase mask.
    phase = np.angle(corrected_image)  # Compute the phase of the corrected image using np.angle.
    phase = phase + np.pi  # Add pi to the phase to shift the values from [-pi, pi] to [0, 2*pi].
    ib = np.where(phase > 0.5, 1, 0)  # Create a binary image ib by thresholding the phase values.
    J = M*N - np.sum(ib)  # Compute the number of pixels with value 0 in the binary image ib.
    return J

def std_CF_noTele_BAR_1d(fun, Cx, holoCompensate, M, N, sign):
    '''
    Variables:
    fun: function to optimize
    Cx: float variable indicating the initial phase factor curvature (seed)
    holoCompensate: hologram data to compensate
    M and N: integer varaibles representing the number of pixels in each dimension of 'holoCompensate'
    '''
    phi_spherical = fun(Cx)  # Compute phi_spherical by calling the function fun with input seed Cx.
    if (sign):
        phase_mask = np.exp((1j)*phi_spherical)  # Compute the phase mask by taking the exponential of 1j times phi_spherical.
    else:
        phase_mask = np.exp((-1j)*phi_spherical)  # Compute the phase mask by taking the exponential of -1j times phi_spherical.
        
    corrected_image = holoCompensate * phase_mask  # Compensate the image by multiplying the hologram with the phase mask.
    phase = np.angle(corrected_image)  # Compute the phase of the corrected image using np.angle.
    phase = phase + np.pi  # Add pi to the phase to shift the values from [-pi, pi] to [0, 2*pi].
    J = np.std(phase)  # Compute the standard deviation of the phase values.
    if Cx == 0:  # If Cx is 0, set J to 0.5 (a special case).
        J = 0.5
    return J  # Return the computed value of J.

def fmincon(minfunc, lb, ub, Cy):
    '''
    Variables:
    minfunc: function to minimize
    lb and ub: float variables indicating low and upper bounderies of the search range
    Cy: float variable indicating the initial phase factor curvature (seed)
    '''

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
    '''
    Variables:
    minfunc: function to minimize
    Cy: float variable indicating the initial phase factor curvature (seed)
    '''
    start = timer()	#Start to count time
    gradient = lambda t: derivative(minfunc, t) #gradient function
    out = fmin_ncg(minfunc, x0 = Cy, fprime=gradient)
    Cy_opt = out
    print("Processing time FMU:", timer()-start) #Time for FMU execution  
    return Cy_opt

def fsolver(minfunc, Cy):
    '''
    Variables:
    minfunc: function to minimize
    Cy: float variable indicating the initial phase factor curvature (seed)
    '''
    start = timer()	#Start to count time
    root = fsolve(minfunc, x0 = Cy)
    Cy_opt = root
    print("Processing time FSO:", timer()-start) #Time for FSO execution
    return Cy_opt

def simulannealbnd(minfunc, lb, ub):
    '''
    Variables:
    minfunc: function to minimize
    lb and ub: float variables indicating low and upper bounderies of the search range
    '''
    start = timer()	#Start to count time
    # Define the bounds of the optimization problem
    bounds = [(lb, ub)]
    # Use the dual_annealing method to minimize the objective function
    out = optimize.dual_annealing(minfunc, bounds)
    Cy_opt = out.x
    print("Processing time SA:", timer()-start) #Time for SA execution  
    return Cy_opt

def pareto_search(minfunc, Cy):
    '''
    Variables:
    minfunc: function to minimize
    Cy: float variable indicating the initial phase factor curvature (seed)
    '''
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
    
def genetic_algorithm(minfunc, lb, ub):
    '''
    Variables:
    minfunc: function to minimize
    lb and ub: float variables indicating low and upper bounderies of the search range
    '''
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
    '''
    Variables:
    minfunc: function to minimize
    Cy: float variable indicating the initial phase factor curvature (seed)
    '''
    start = timer()	#Start to count time
    out = optimize.minimize(fun = minfunc, x0 = Cy, method='Nelder-Mead')
    if out.success:
        print("Optimization successful!")
    else:
        print("Optimization failed.")
    Cy_opt = out.x
    print("Processing time PS:", timer()-start) #Time for PS execution
    return Cy_opt

def hybrid_ga_ps(minfunc, lb, ub):
    '''
    Variables:
    minfunc: function to minimize
    lb and ub: float variables indicating low and upper bounderies of the search range
    '''
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

    
# Functions for phase compensation using manually input parameters
def spatialFilterinCNT(inp, M, N):
    ROI_array = np.zeros(4)
    holoFT = np.float32(inp)  # convertion of data to float
    fft_holo = cv2.dft(holoFT, flags=cv2.DFT_COMPLEX_OUTPUT)  # FFT of hologram
    fft_holo = np.fft.fftshift(fft_holo)
    fft_holo_image = 20 * np.log(cv2.magnitude(fft_holo[:, :, 0], fft_holo[:, :, 1]))  # logaritm scale FFT
    minVal = np.amin(np.abs(fft_holo_image))
    maxVal = np.amax(np.abs(fft_holo_image))
    fft_holo_image = cv2.convertScaleAbs(fft_holo_image, alpha=255.0 / (maxVal - minVal),
                                         beta=-minVal * 255.0 / (maxVal - minVal))

    ROI = cv2.selectROI(fft_holo_image, fromCenter=True)  # module to  ROI
    # imCrop = fft_holo_image[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]):int(ROI[0] + ROI[2])]
    x1_ROI = int(ROI[1])
    y1_ROI = int(ROI[0])
    x2_ROI = int(ROI[1] + ROI[3])
    y2_ROI = int(ROI[0] + ROI[2])
    ROI_array[0] = x1_ROI
    ROI_array[1] = y1_ROI
    ROI_array[2] = x2_ROI
    ROI_array[3] = y2_ROI

    # computing the center of the rectangle mask
    Ycenter = x1_ROI + (x2_ROI - x1_ROI)/2
    Xcenter = y1_ROI + (y2_ROI - y1_ROI)/2

    holo_filter = np.zeros((M, N, 2))
    holo_filter[x1_ROI:x2_ROI, y1_ROI: y2_ROI] = 1
    holo_filter = holo_filter * fft_holo
    holo_filter = np.fft.ifftshift(holo_filter)
    holo_filter = cv2.idft(holo_filter, flags=cv2.DFT_INVERSE)

    holo_filter_real = holo_filter[:, :, 0]
    holo_filter_imag = holo_filter[:, :, 1]
    holo_filter = np.zeros((M, N), complex)
    for p in range(M):
        for q in range(N):
            holo_filter[p, q] = complex(holo_filter_real[p, q], holo_filter_imag[p, q])

    return Xcenter, Ycenter, holo_filter, ROI_array

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONUP:
        mouse_x, mouse_y = x, y
        print('Pixel coordinates selected:', x, y)

def brute_search(comp_phase, arrayCurvature, arrayXcenter, arrayYcenter, wavelength, X, Y, dx, dy, sign, vis):
    
    sum_max = 0
    cont = 0
    
    for curTemp in arrayCurvature:
        #print ("... in 'cur': ", curTemp, " going to: ", arrayCurvature[-1])
        for gTemp in arrayXcenter:
            for hTemp in arrayYcenter:
                cont = cont + 1
                
                phaseCompensate, phi_spherical, sum = TSM(comp_phase, curTemp, gTemp, hTemp, wavelength, X, Y, dx, dy, sign)
                
                if (sum > sum_max):
                    g_out = gTemp
                    h_out = hTemp
                    cur_out = curTemp
                    sum_max = sum
                    
                    if (vis):
                        # Create a figure with three subplots
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                
                        # Plot the first image in the first subplot
                        ax1.imshow(np.angle(phi_spherical))
                        ax1.set_title('phi_spherical')
                
                        # Plot the second image in the second subplot
                        ax2.imshow(np.angle(comp_phase))
                        ax2.set_title('comp_phase')
                
                        # Plot the third image in the third subplot
                        ax3.imshow(phaseCompensate)
                        ax3.set_title('phaseCompensate')
                
                        # Show the figure
                        plt.show()
                    
    return g_out, h_out, cur_out, sum_max

def TSM(comp_phase, curTemp, gTemp, hTemp, wavelength, X, Y, dx, dy, sign):
    phi_spherical = (np.power(X - gTemp, 2) * np.power(dx, 2) / curTemp) + (np.power(Y - hTemp, 2) * np.power(dy, 2) / curTemp)
    phi_spherical = math.pi * phi_spherical / wavelength
    if (sign == True):
        phi_spherical = np.exp(1j * phi_spherical)
    else:
        phi_spherical = np.exp(-1j * phi_spherical)
                
    phaseCompensate = comp_phase * phi_spherical
    phaseCompensate = np.angle(phaseCompensate)
    
    minVal = np.amin(phaseCompensate)
    maxVal = np.amax(phaseCompensate)
    phase_sca = (phaseCompensate - minVal) / (maxVal - minVal)
    binary_phase = (phase_sca > 0.2)

    # Applying the summation and thresholding metric
    sum = np.sum(np.sum(binary_phase))

    return phaseCompensate, phi_spherical, sum                  

def fast_CNT(inp, wavelength, dx, dy):
    '''
    # Function for rapid compensation of phase maps of image plane off-axis DHM, operating in non-telecentric regimen
    # Inputs:
    # inp - The input intensity (captured) hologram
    # wavelength - Wavelength of the illumination source to register the DHM hologram
    # dx, dy - Pixel dimensions of the camera sensor used for recording the hologram
    '''

    #wavelength = wavelength * 0.000001
    #dx = dx * 0.000001
    #dy = dy * 0.000001

    # Retrieving the input shape
    inp = np.array(inp)
    M, N = inp.shape
    k = (2 * math.pi) / wavelength

    # Creating a mesh-grid to operate in world coordinates
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')  # meshgrid XY

    # The spatial filtering process is executed
    print("Spatial filtering process started.....")
    Xcenter, Ycenter, holo_filter, ROI_array = spatialFilterinCNT(inp, M, N)
    print("Spatial filtering process finished.")

    # Fourier transform to the hologram filtered
    ft_holo = FT(holo_filter)
    #plt.figure(); plt.imshow(np.abs(ft_holo)**2, cmap='gray'); plt.title('FT Filtered holo'); 
    #plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    # reference wave for the first compensation (global linear compensation)
    ThetaXM = math.asin((N / 2 - Xcenter) * wavelength / (N * dx))
    ThetaYM = math.asin((M / 2 - Ycenter) * wavelength / (M * dy))
    reference = np.exp(1j * k * (math.sin(ThetaXM) * X * dx + math.sin(ThetaYM) * Y * dy))

    # First compensation (tilting angle compensation)
    comp_phase = holo_filter * reference
    phase_c = np.angle(comp_phase)

    # show the first compensation
    minVal = np.amin(phase_c)
    maxVal = np.amax(phase_c)
    phase_normalized = (phase_c - minVal) / (maxVal - minVal)
    binary_phase = (phase_normalized > 0.5)
    #plt.figure(); plt.imshow(binary_phase, cmap='gray'); plt.title('Binarized phase'); 
    #plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    # creating the new reference wave to eliminate the circular phase factors (second phase compensation)
    m = abs(ROI_array[2] - ROI_array[0])
    n = abs(ROI_array[3] - ROI_array[1])
    Cx = np.power((M * dx), 2)/(wavelength * m)
    Cy = np.power((N * dy), 2)/(wavelength * n)
    cur = (Cx + Cy)/2
    
    print ("Select the center of the spherical phase factor and press 'esc'")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    cv2.imshow('image', phase_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Get the mouse click coordinates from the global variables
    p, q = mouse_x, mouse_y
    
    print ("p & q: ", p, q)
    
    g = ((M/2) - int(p))/2
    h = ((N/2) - int(q))/2
    
    #Let's test the sign of the spherical phase factor
    
    s = max(M,N)*0.05
    step = max(M,N)*0.1
    perc = 0.05
    arrayCurvature = np.arange(cur - (cur*perc), cur + (cur*perc), perc/2)
    arrayXcenter = np.arange(g - s, g + s, step)
    arrayYcenter = np.arange(h - s, h + s, step)
    
    sign = True
    sum_max_True = brute_search(comp_phase, arrayCurvature, arrayXcenter, arrayYcenter, wavelength, X, Y, dx, dy, sign, vis = False)[3]
    #print (sum_max_True)
    
    sign = False
    sum_max_False = brute_search(comp_phase, arrayCurvature, arrayXcenter, arrayYcenter, wavelength, X, Y, dx, dy, sign, vis = False) [3]
    #print (sum_max_False)
    
    if (sum_max_True > sum_max_False):
        sign = True
    else:
        sign = False
        
    #sign = True
        
    print ("Sign of spherical phase factor: ", sign)
    
    step_g = max(M,N)*0.1
    step_h = step_g
    step_cur = 0.6*cur
    
    current_point = (cur, g, h)
    
    phaseCompensate, phi_spherical, current_value = TSM(comp_phase, current_point[0], current_point[1], current_point[2], wavelength, X, Y, dx, dy, sign)
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                
    # Plot the first image in the first subplot
    ax1.imshow(np.angle(phi_spherical))
    ax1.set_title('phi_spherical')
                
    # Plot the second image in the second subplot
    ax2.imshow(np.angle(comp_phase))
    ax2.set_title('comp_phase')
                
    # Plot the third image in the third subplot
    ax3.imshow(phaseCompensate)
    ax3.set_title('phaseCompensate')
    
    # Show the figure
    plt.show()

    while True:
        
        neighbors = []
        for dex in [-2, -1, 0, 1, 2]:
            for dey in [-2, -1, 0, 1, 2]:
                for dez in [-2, -1, 0, 1, 2]:
                    if dex == dey == dez == 0:
                        continue
                    neighbor = (current_point[0] + dez*step_cur, current_point[1] + dex*step_g, current_point[2] + dey*step_h)
                    neighbors.append(neighbor)
        #print ("current_point", current_point)
        #print ("neighbors", neighbors)
        optimal_neighbor = current_point
        optimal_value = current_value
        for neighbor in neighbors:
            #print ("neighbor: ",neighbor)
            phaseCompensate, phi_spherical, neighbor_value = TSM(comp_phase, neighbor[0], neighbor[1], neighbor[2], wavelength, X, Y, dx, dy, sign)
            
            if neighbor_value > optimal_value:
                optimal_neighbor = neighbor
                optimal_value = neighbor_value
                print ("Best neighbor found! ", optimal_value)
                # Create a figure with three subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                
                # Plot the first image in the first subplot
                ax1.imshow(np.angle(phi_spherical))
                ax1.set_title('phi_spherical')
                
                # Plot the second image in the second subplot
                ax2.imshow(np.angle(comp_phase))
                ax2.set_title('comp_phase')
                
                # Plot the third image in the third subplot
                ax3.imshow(phaseCompensate)
                ax3.set_title('phaseCompensate')
                
                # Show the figure
                plt.show()
        
        #if np.abs(optimal_value - current_value) < optimal_value*0.00005:
        #    break
        #el
        if optimal_value > current_value:
            current_point = optimal_neighbor
            current_value = optimal_value
        else:
            break
        
        step_g = step_g*0.5
        step_h = step_g
        step_cur = step_cur*0.5
    
    #current_point #(cur, f, g)
    
    phi_spherical = (np.power(X - current_point[1], 2) * np.power(dx, 2) / current_point[0]) + (np.power(Y - current_point[2], 2) * np.power(dy, 2) / current_point[0])
    phi_spherical = math.pi * phi_spherical / wavelength
    
    if (sign == True):
        phi_spherical = np.exp(1j * phi_spherical)
    else:
        phi_spherical = np.exp(-1j * phi_spherical)

    phaseCompensate = comp_phase * phi_spherical

    print("Phase compensation finished.")

    return phaseCompensate