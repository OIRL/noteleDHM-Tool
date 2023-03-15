"""
Title: no_tele_main.py

Description: This Python script provides phase compensation for DHM holograms recorded in a non-telecentric configuration. 
The code was originally developed in MATLAB by Brian Bogue-Jimenez, with assistance from Prof. Ana Doblas and PhD student 
Raul Castañeda. This Python version of the code has been fully automated, requiring only that the user input the pixel size 
in each dimension (dx and dy) and the illumination wavelength (Lambda).

Dependencies: The code uses functions from the 'funs.py' script.

Date: January 6, 2023
Last update: March 10, 2023.

Authors: Brian Bogue-Jimenez, Carlos Trujillo, and Ana Doblas
"""

#All operations are perfomed using numpy
import numpy as np
#Plotting figures
import matplotlib.pyplot as plt

#library to count procesing time
from timeit import default_timer as timer

#Import of the needed functions
import funs

'''
The main code starts here
'''

#Different image files to process (no-tele holograms). Sample input holograms located in 'data/'
#["4cm_20x_bigcakes.tiff", "-4cm_20x_star.tiff", "4cm_20x_usaf.tiff", "RBCnotele50x.tiff", "RBCnotele50x_square.tiff"]
#["4cm_20x_usaf_square.tiff"]

#Loading image file (hologram) to process
user_input = '-4cm_20x_star.tiff'
filename = 'data/' + user_input
print ('Non-telecentric DHM hologram: ', filename)

vargin = 1 #Scalling factor of the input images
holo, M, N, X, Y = funs.holo_read(filename, vargin)

plt.figure(); plt.imshow(holo, cmap='gray'); plt.title('Hologram'); 
plt.gca().set_aspect('equal', adjustable='box'); plt.show()

#Variables and flags for hologram reconstruction (Default variables, change accordingly)
#Lambda = 532*10**(-9)
Lambda = 633*10**(-9)
k = 2*np.pi/Lambda
dx = 6.9*10**(-6)
dy = 6.9*10**(-6)

print ('Phase compensation starts...')

'''
0: Manual determination of the M&N and H&G coordinates for no-tele compensation. 
1: Automatic determination of these parameters.
'''
auto = 0

if auto:

    #Let's go to the spatial frequency domain
    FT_holo = funs.FT(holo);

    #Let's threshold that FT
    BW = funs.threshold_FT(FT_holo, M, N)

    #Get the +1 D.O. term region and coordinates
    #start = timer()	#Start to count time
    plus_coor, m, n, p, q = funs.get_plus1(BW)
    #print("Processing time get_plus1:", timer()-start) #Time for get_plus1 execution

    #Compensating the tilting angle first
    off = 0 #some kind of offset
    holoCompensate = funs.filter_center_plus1(FT_holo,plus_coor,m-off,n-off,Lambda,X,Y,dx,dy,k)

    # Binarized Spherical Aberration
    BW = funs.binarize_compensated_plus1(holoCompensate)

    # Get the center of the remaining spherical phase factor for the 2nd compensation
    g, h = funs.get_g_and_h(BW)
    
else: 

    #Compensating the tilting angle first (Manual)
    #holoCompensate, m, n = funs.filter_center_plus1_manual(holo,Lambda,X,Y,dx,dy,k)
    
    start = timer()	#Start to count time
    
    # Numerical compensation using the CNT approach
    output = funs.CNT(holo, Lambda, dx, dy, spatialFilter='sfmr')
    plt.figure(); plt.imshow(np.angle(output), cmap='gray'); plt.title('Nested loops optimized compensated image'); 
    plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    print("Processing time CNT:", timer()-start) #Time for CNT execution    
    
    # Get the center of the remaining spherical phase factor for the 2nd compensation manually
    #g, h = funs.get_g_and_h_manual(holoCompensate)

#Let's create the new reference wave to eliminate the circular phase factors. 
Cx = np.power((M * dx), 2)/(Lambda * m)
Cy = np.power((N * dy), 2)/(Lambda * n)
cur = (Cx + Cy)/2

#Let's built the spherical phase factor for compensation
phi_spherical = funs.phi_spherical_C(cur, g, h, dx, X, Y, Lambda)
phase_mask = np.exp((-1j)*phi_spherical)
phase_mask_2 = np.exp((1j)*phi_spherical) #Negative curvature

#Let's apply the second (quadratic) compensation according to Kemper
corrected_image = holoCompensate * phase_mask
    
plt.figure(); plt.imshow(np.angle(corrected_image), cmap='gray'); plt.title('Non-optimized compensated image'); 
plt.gca().set_aspect('equal', adjustable='box'); plt.show()

#Let's apply the second (quadratic) compensation according to Kemper (negative phase factor)
corrected_image = holoCompensate * phase_mask_2
    
plt.figure(); plt.imshow(np.angle(corrected_image), cmap='gray'); plt.title('Non-optimized compensated image (Negative curvature)'); 
plt.gca().set_aspect('equal', adjustable='box'); plt.show()


'''
Up to this point the phase compensation for no telecentric DHM holograms is finished according to Kemper
'''

# Set the default random number generator for reproducibility
np.random.seed(0)

#Different available optimization methods
alg_array = ["FMC","FMU","FSO","SA","PTS","GA","PS","GA+PS", "BRUTE"]
#0: FMC 1: FMU 2: FSO 3: SA 4: PTS 5: GA 6: PS 7: GA+PS 8: BRUTE  (See documentation for further details)
i = 5; #Select method as desired
alg = alg_array[i]

#Two available const functions
cost_fun = ['BIN cost function','STD cost function']
#cost = 1 # 0 - BIN -- 1 - SD (See documentation for further details)
cost = 0 
print ('Selectec cost function: ', cost_fun[cost])

# Define the function phi_spherical_C for the optimization (it's the same used before, but built for optimization)
phi_spherical_C = lambda C: (np.pi / (C * Lambda)) * ((X - (g + 1))**2 + (Y - (h + 1))**2) * (dx**2)

# Set the cost function
if cost == 0:
    minfunc = lambda t: funs.bin_CF_noTele_BAR_1d(phi_spherical_C, t, holoCompensate, M, N)
elif cost == 1:
    minfunc = lambda t: funs.std_CF_noTele_BAR_1d(phi_spherical_C, t, holoCompensate, M, N)
  
#Determination (minimization) of the optimal parameter (C -curvature) for the accurate phase compensation of no tele DHM holograms.

# Define the lower and upper bounds of the initial population range (Warning: Modifying these settings may cause unexpected behavior. Proceed with caution)
lb = -0.5
ub = 0.5
lb = cur + cur * lb
ub = cur + cur * ub

print ('cur :', cur)
print ('lb and ub: ', lb, ub)

# Minimize the cost function using the selected algorithm
if alg == "GA+PS":
    print ('Running the hybrid GA + PS strategy with the', cost_fun[cost])
    Cy_opt = funs.hybrid_ga_ps(minfunc, lb, ub)
elif alg == "GA":
    print ('Running the genetic algorithm with the', cost_fun[cost])
    Cy_opt = funs.genetic_algorithm(minfunc, lb, ub)
elif alg == "PS":
    print ('Running the pattern search algorithm with the', cost_fun[cost])
    Cy_opt = funs.pattern_search(minfunc, cur)
elif alg == "FMC": #fmincon via 'NonlinearConstraint'
    print ('Running the NonlinearConstraint (fmincon) algorithm with the', cost_fun[cost])
    Cy_opt = funs.fmincon(minfunc, lb, ub, cur)
elif alg == "FMU":  #fminunc via fmin_ncg
    print ('Running the fmin_ncg (fminunc) algorithm with the', cost_fun[cost])
    Cy_opt = funs.fminunc(minfunc, cur)
elif alg == "FSO": #fsolve
    print ('Running the function solver (fsolver) algorithm with the', cost_fun[cost])
    Cy_opt = funs.fsolver(minfunc, cur)
elif alg == "PTS": #paretosearch
    print ('Running the pareto search strategy with the', cost_fun[cost])
    Cy_opt = funs.pareto_search(minfunc, cur)
elif alg == "PSW": #particleswarm
    print ('Running the particle swarm optimization with the', cost_fun[cost])
    Cy_opt = funs.particleswarm(minfunc, lb, ub)
elif alg == "SA": #simulannealbnd
    print ('Running the simulated annualing algortihm with the', cost_fun[cost])
    Cy_opt = funs.simulannealbnd(minfunc, lb, ub)
elif alg == "SGO": #surrogateopt
    print ('Running the surrogate optimization with the', cost_fun[cost])
    Cy_opt = funs.surrogateopt(minfunc, lb, ub)
elif alg == "BRUTE": #surrogateopt
    print ('Running the brute force optimization with the', cost_fun[cost])
    Cy_opt = funs.brute(minfunc, lb, ub)
else:
    print('No proper optimization method selected')

#Let's compute the optimized compensation phase factor
phi_spherical = phi_spherical_C(Cy_opt)

print ('C optimized: ', Cy_opt)

phase_mask = np.exp((-1j)*phi_spherical) #complex matrix of the compensating quadratic phase factor
corrected_image = holoCompensate * phase_mask #Compensation
plt.figure(); plt.imshow(np.angle(corrected_image), cmap='gray'); plt.title('Corrected_image'); 
plt.gca().set_aspect('equal', adjustable='box'); plt.show()
