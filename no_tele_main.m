%{
Title: no_tele_main.m

Description: This matlab script provides phase compensation for DHM holograms recorded in a non-telecentric configuration. 
The code was originally developed by Brian Bogue-Jimenez, with assistance from Prof. Ana Doblas and Dr. Raúl Castañeda.
This matlab version of the code has been fully automated, requiring only that the user input the pixel size 
in each dimension (dx and dy) and the illumination wavelength (Lambda).

Dependencies: The code uses functions from the 'funs.m' script.

Date: January 6, 2023
Last update: March 24, 2023.

Authors: Brian Bogue-Jimenez, Carlos Trujillo, and Ana Doblas
%}

% clear memory 

clc % clean
close all% close all windows
clear all% clear of memory all variable

% The main code starts here

% Different image files to process (no-tele holograms). Sample input holograms located in 'data/'
% ["4cm_20x_bigcakes.tiff", "-4cm_20x_star.tiff", "4cm_20x_usaf.tiff", "RBCnotele50x.tiff"]

% Loading image file (hologram) to process
user_input = '4cm_20x_usaf.tiff';
filename = ['data/', user_input];
disp(['Non-telecentric DHM hologram: ', filename]);

vargin = 1; % Scalling factor of the input images
[holo, M, N, X, Y] = funs.holo_read(filename, vargin);

figure,imagesc(holo),colormap(gray),title('Hologram'),daspect([1 1 1])

% Variables and flags for hologram reconstruction (Default variables, change accordingly)
Lambda = 633*10^(-9);
dx = 6.9*10^(-6);
dy = 6.9*10^(-6);

%Different available optimization methods (only meeded for automatic method)
%1: FMC 2: FMU 3: FSO 4: SA 5: PTS 6: GA 7: PS 8: GA+PS  (See documentation for further details)
algo = 8; %Select method as desired

%Two available cost functions (only needed for automatic method)
%cost = 1 # 0 - BIN -- 1 - SD (See documentation for further details)
cost = 1; %Select function as desired

disp('Phase compensation starts...');

%{
0: Manual determination of the M&N and H&G coordinates for no-tele compensation. 
1: Automatic determination of these parameters.
%}

auto = 0;

if auto
    
    %# Numerical compensation using an automatic method to determine the +1 ROI and center of the spherical phase factor
    tic
    out = funs.automatic_method(holo, M, N, X, Y, Lambda, dx, dy, algo, cost);
    toc

    figure; colormap gray; imagesc(angle(out));
    title('Compensated image after automatic optimization');axis square

else
    
    % Numerical compensation using a semiheuristic version of the CNT approach from pyDHM
    tic
    out = funs.fast_CNT(holo, Lambda, M, N, X, Y, dx, dy);
    toc

    figure; colormap gray; imagesc(angle(out));
    title('Semiheuristically optimized compensated image');axis square

end