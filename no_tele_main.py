"""
Title: no_tele_main.py

Description: This Python script provides phase compensation for DHM holograms recorded in a non-telecentric configuration. 
The code was originally developed in MATLAB by Brian Bogue-Jimenez, with assistance from Prof. Ana Doblas and PhD student 
Raul Castañeda. This Python version of the code has been fully automated, requiring only that the user input the pixel size 
in each dimension (dx and dy) and the illumination wavelength (Lambda).

Dependencies: The code uses functions from the 'funs.py' script.

Date: January 6, 2023
Last update: March 24, 2023.

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
#["4cm_20x_bigcakes.tiff", "-4cm_20x_star.tiff", "4cm_20x_usaf.tiff", "RBCnotele50x.tiff"]

#Loading image file (hologram) to process
user_input = '4cm_20x_usaf.tiff'
filename = 'data/' + user_input
print ('Non-telecentric DHM hologram: ', filename)

vargin = 1 #Scalling factor of the input images
holo, M, N, X, Y = funs.holo_read(filename, vargin)

plt.figure(); plt.imshow(holo, cmap='gray'); plt.title('Hologram'); 
plt.gca().set_aspect('equal', adjustable='box'); plt.show()

#Variables and flags for hologram reconstruction (Default variables, change accordingly)
#Lambda = 532*10**(-9)
Lambda = 633*10**(-9)
dx = 6.9*10**(-6)
dy = 6.9*10**(-6)

#Different available optimization methods (only meeded for automatic method)
#0: FMC 1: FMU 2: FSO 3: SA 4: PTS 5: GA 6: PS 7: GA+PS  (See documentation for further details)
algo = 7; #Select method as desired

#Two available cost functions (only needed for automatic method)
#cost = 1 # 0 - BIN -- 1 - SD (See documentation for further details)
cost = 1 #Select function as desired

print ('Phase compensation starts...')

###################################################################################
#0: Manual determination of the M&N and H&G coordinates for no-tele compensation.## 
#1: Automatic determination of these parameters. ##################################
###################################################################################
auto = 1

if auto:

    start = timer()	#Start to count time
    
    # Numerical compensation using an automatic method to determine the +1 ROI and center of the spherical phase factor
    output = funs.automatic_method(holo, M, N, X, Y, Lambda, dx, dy, algo, cost)
    plt.figure(); plt.imshow(np.angle(output), cmap='gray'); plt.title('Compensated imaged after optimization'); 
    plt.gca().set_aspect('equal', adjustable='box'); plt.show()

    print("Processing time automatic method:", timer()-start) #Time for CNT execution   

else: 
    
    start = timer()	#Start to count time
    
    # Numerical compensation using a semiheuristic version of the CNT approach from pyDHM
    output = funs.fast_CNT(holo, Lambda, dx, dy)
    plt.figure(); plt.imshow(np.angle(output), cmap='gray'); plt.title('Semiheuristically optimized compensated image'); 
    plt.gca().set_aspect('equal', adjustable='box'); plt.show()
    
    print("Processing time fastCNT:", timer()-start) #Time for CNT execution    
