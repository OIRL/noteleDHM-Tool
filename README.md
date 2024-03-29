# noteleDHM-Tool

noteleDHM is an implementation developed in collaboration between the Optical Imaging Research Laboratory in the Department of Electrical and Computer Engineering at the University of Memphis and the Applied Optics Research Group in the School of Applied Sciences and Engineering at Universidad EAFIT in Medellin, Colombia. The noteleDHM MATLAB App is a free, user-friendly tool that provides a streamlined strategy for reconstructing off-axis holograms recorded using a digital holographic microscope (DHM) operating in non-telecentric mode. The implementation is based on a spectral method [1] in combination with a minimization algorithm that finds the parameters of the spherical wavefront to reconstruct a quantitative phase image with minimal phase perturbations due to the interference angle of the off-axis configuration and the non-telecentric configuration in the DHM imaging system.


The noteleDHM method requires the sensor's pixel size and the wavelength of the light source as input parameters. The algorithm provides an open-source reconstruction tool for the DHM community, allowing researchers to accurately reconstruct quantitative phase images from off-axis holograms with minimal phase perturbations.

# Documentation

Full documentation and a user manual of noteleDHM can be found in:

https://sites.google.com/view/noteledhmtool/home?authuser=0&pli=1


# Requirements

To install the required libraries for running the noteleDHM-toll Python scripts in a Conda environment on Windows, follow these steps:

1. Open the command prompt or Anaconda prompt.
  
2. Type the following commands to install the necessary libraries:

    conda install -c conda-forge numpy matplotlib

    conda install -c conda-forge opencv

    conda install -c conda-forge imageio

    conda install -c conda-forge scikit-image

    Alternatively, you can also install all of the libraries using 'pip'.

# Credits

•	noteleDHM is developed in MATLAB 2021a (version 9.19.0, R2021a, Natick, Massachusetts: The MathWorks Inc.) and Python 3.7.1 (2018).

•	For the unwrapping step in the MATLAB GUI, nonteleDHM implements the code developed by Herráez et.al. [2,3]

## References 

[1] J. Min, B. Yao, S. Ketelhut, C. Engwer, B. Greve, B. Kemper, Opt. Lett. 42 (2017) 227 - 230. https://doi.org/10.1364/OL.42.000227.

[2] M. A. Herraez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, "Fast two-dimensional phase-  unwrapping algorithm based on sorting by reliability following a noncontinuous path", Applied Optics, Vol. 41, Issue 35, pp. 7437-7444 (2002).

[3] M. F. Kasim, "Fast 2D phase unwrapping implementation in MATLAB", available in https://github.com/mfkasim91/unwrap_phase/.

# Citation

If using noteleDHM for publication, please kindly cite the following:

•	B. Bogue-Jimenez, C. Trujillo, and A. Doblas, “Comprehensive Tool for a Phase Compensation Reconstruction Method in Digital Holographic Microscopy Operating in Non-Telecentric Regime,” Computer Physics Communications, under review (2023).

# Support

If you use noteleDHM and find a bug, please contact us via email and we will address the problem. Our emails are:

opticalimagingresearchlab@gmail.com

catrujilla@eafit.edu.co

adoblas@memphis.edu


