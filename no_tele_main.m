%{
Title: no_tele_main.m

Description: This matlab script provides phase compensation for DHM holograms recorded in a non-telecentric configuration. 
The code was originally developed by Brian Bogue-Jimenez, with assistance from Prof. Ana Doblas and Dr. Raúl Castañeda.
This matlab version of the code has been fully automated, requiring only that the user input the pixel size 
in each dimension (dx and dy) and the illumination wavelength (Lambda).

Dependencies: The code uses functions from the 'funs.m' script.

Date: January 6, 2023
Last update: March 23, 2023.

Authors: Brian Bogue-Jimenez, Carlos Trujillo, and Ana Doblas
%}

%% clear memory 

clc % clean
close all% close all windows
clear all% clear of memory all variable

% The main code starts here

% Different image files to process (no-tele holograms). Sample input holograms located in 'data/'
% ["4cm_20x_bigcakes.tiff", "-4cm_20x_star.tiff", "4cm_20x_usaf.tiff", "RBCnotele50x.tiff"]

% Loading image file (hologram) to process
user_input = '4cm_20x_bigcakes.tiff';
filename = ['data/', user_input];
disp(['Non-telecentric DHM hologram: ', filename]);

vargin = 1; % Scalling factor of the input images
[holo, M, N, X, Y] = funs.holo_read(filename, vargin);

figure,imagesc(holo),colormap(gray),title('Hologram'),daspect([1 1 1])

% Variables and flags for hologram reconstruction (Default variables, change accordingly)
Lambda = 633*10^(-9);
k = 2*pi/Lambda;
dx = 6.9*10^(-6);
dy = 6.9*10^(-6);

disp('Phase compensation starts...');

%{
0: Manual determination of the M&N and H&G coordinates for no-tele compensation. 
1: Automatic determination of these parameters.
%}

auto = 1;

if auto
    
    %Let's go to the spatial frequency domain
    FT_holo = funs.FT(holo);
    %figure,imagesc(log(abs(FT_holo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) ;

    %Let's threshold that FT
    BW = funs.threshold_FT(FT_holo, M, N);

    %Get the +1 D.O. term region and coordinates
    %tic
    [plus_coor,m,n,p,q] = funs.get_plus1(BW);
    %toc

    %Compensating the tilting angle first
    holoCompensate = funs.filter_center_plus1(FT_holo,plus_coor,m,n,Lambda,X,Y,dx,dy,k);
    figure,imagesc(angle(holoCompensate)),colormap(gray),title('IFT Hologram'),daspect([1 1 1]);
    
    %Binarized Spherical Aberration
    bw = funs.binarize_compensated_plus1(holoCompensate);
    figure,imshow(bw),colormap(gray),title('Binarized Spherical Aberration'),daspect([1 1 1]) 

    %Get the center of the remaining spherical phase factor for the 2nd compensation
    [g,h] = funs.get_g_and_h(bw);

    %%Let's create the new reference wave to eliminate the circular phase factors
    phi_spherical_C = @(C) pi/Lambda*((X-(g)+ 1).^2 + (Y-(h)+1).^2)*(dx^2)/(C);
    
    Cx = (M*dx)^2 / (Lambda*m); 
    Cy = (N*dy)^2 / (Lambda*n);
    cur = (Cx + Cy)/2;

    %Let's select the sign of the compensating spherical phase factor
    sign = 1;
    
    %Let's built the spherical phase factor for compensation
    phi_spherical = phi_spherical_C(cur);

    if sign
        phase_mask = exp((1j)*phi_spherical);
    else
        phase_mask = exp((-1j)*phi_spherical);
    end

    corrected_image = holoCompensate .* phase_mask;
    figure; colormap gray; imagesc(angle(corrected_image));
    title('Non-optimized compensated image');axis square
            
    % Up to this point the phase compensation for no telecentric DHM holograms is finished according to Kemper
    
    %Set the default random number generator for reproducibility
    rng(0);
    
    %Different available optimization methods
    alg_array = ["FMC","FMU","FSO","SA","PTS","GA","PS","GA+PS"];
    %1: FMC 2: FMU 3: FSO 4: SA 5: PTS 6: GA 7: PS 8: GA+PS  (See documentation for further details)
    i = 7; %Select method as desired
    alg = alg_array(i);

    % Two available cost functions
    cost_fun = {'BIN cost function', 'STD cost function'};
    % cost = 0 for BIN, 1 for SD (See documentation for further details)
    cost = 1;
    disp(['Selected cost function: ', cost_fun{cost+1}])

    %Define the function phi_spherical_C for the optimization (it's the same used before, but built for optimization)
    phi_spherical_C = @(C) (pi/(C*Lambda))*((X-(g)+ 1).^2 + (Y-(h)+1).^2)*(dx^2);
    
    if cost == 0
        minfunc = @(t)funs.bin_CF_noTele_BAR_1d(phi_spherical_C,t,holoCompensate,M,N);
    elseif cost == 1 
        minfunc = @(t)funs.std_CF_noTele_BAR_1d(phi_spherical_C,t,holoCompensate,M,N);
    end  %select cost func

    %Determination (minimization) of the optimal parameter (C -curvature) for the accurate phase compensation of no tele DHM holograms.
    
    %Define the lower and upper bounds of the initial population range (Warning: Modifying these settings may cause unexpected behavior. Proceed with caution)
    lb = @(C)-.5;ub = @(C).5; % +-C*50%
    lb = lb(cur); ub = ub(cur);
    x0 = cur;
    nvars = 1;
    
    disp(['cur: ', num2str(cur)])
    disp(['lb and ub: ', num2str(lb), ', ', num2str(ub)])

    % Options for the Optimization functions
    if alg == "GA"
        init_pop_range_1d = @(C, lb, ub) [C+C*lb;C+C*ub];
        options = optimoptions(@ga,...
        'InitialPopulationMatrix', x0,...
        'InitialPopulationRange',  init_pop_range_1d(x0, lb, ub), ...
        'PopulationSize',15, ... % this speeds up convergence but not needed, default is 50 when nvars<=5
        'SelectionFcn','selectionremainder');
    elseif alg == "GA+PS"
        init_pop_range_1d = @(C, lb, ub) [C+C*lb;C+C*ub];
        hybridopts = optimoptions('patternsearch','Display','final');
        options = optimoptions('ga', 'Display', 'final','HybridFcn', {@patternsearch,hybridopts},...
        'InitialPopulationRange', init_pop_range_1d(x0, lb, ub),...
        'PopulationSize',15, ... % this speeds up convergence but not needed
        'SelectionFcn','selectionremainder');
    elseif alg == "PS"
        options = [];
    elseif alg == "FMC"
        options = optimoptions("fmincon");
    elseif  alg == "PTS"
        options = optimoptions("paretosearch");
    elseif alg == "FMU"
        options = optimoptions("fminunc");
    elseif alg == "FSO"
        options = optimoptions("fsolve");
    elseif alg == "SA"
        options = optimoptions("simulannealbnd");
    end  
    
    % Minimize the cost function using the selected algorithm
    substr = alg.split('+');

    if substr(1) == "GA"
        disp(['Running the genetic algorithm with the ', cost_fun{cost+1}])
        [out,costs] = ga(minfunc,nvars,[],[],[],[],[],[],[],options);
    elseif alg == "PS"
        disp(['Running the pattern search algorithm with the ', cost_fun{cost+1}])
        [out,costs] = patternsearch(minfunc,[x0],[],[],[],[],[],[],[],options);
    elseif alg == "FMC"
        disp(['Running the fmincon algorithm with the ', cost_fun{cost+1}])
        [out,costs] = fmincon(minfunc,x0,[],[],[],[],lb,ub,[],options);
    elseif alg == "FMU"
        disp(['Running the fminunc algorithm with the ', cost_fun{cost+1}])
        [out,costs] = fminunc(minfunc,x0,options);
    elseif alg == "FSO"
        disp(['Running the fsolver algorithm with the ', cost_fun{cost+1}])
        [out,costs] = fsolve(minfunc,x0,options);
    elseif alg == "SA"
        disp(['Running the simulated annualing algorithm with the ', cost_fun{cost+1}])
        [out,costs] = simulannealbnd(minfunc,x0,lb,ub,options);
    elseif alg == "PTS"
        disp(['Running the pareto search algorithm with the ', cost_fun{cost+1}])
        [out,costs] = paretosearch(minfunc,nvars,[],[],[],[],lb,ub,[],options);
    else
        disp('No proper optimization method selected')
    end 

    disp(['C optimized: ', num2str(out)])

    %Let's compute the optimized compensation phase factor
    phi_spherical = phi_spherical_C(out);

    if sign
        phase_mask = exp((1j)*phi_spherical);
    else
        phase_mask = exp((-1j)*phi_spherical);
    end
    
    corrected_image = holoCompensate .* phase_mask;
    figure; colormap gray; imagesc(angle(corrected_image));
    title('Compensated image after optimization');axis square

else

    
end