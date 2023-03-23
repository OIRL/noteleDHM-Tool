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
    sign = 0;
    
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
    i = 5; %Select method as desired
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
        hybridopts = optimoptions('hybrid','Display','final');
        options = optimoptions('ga', 'Display', 'final','HybridFcn', {@patternsearch,hybridopts},...
        'InitialPopulationRange', init_pop_range_1d(x0, lb, ub),...
        'PopulationSize',15, ... % this speeds up convergence but not needed
        'SelectionFcn','selectionremainder');
    elseif alg == "PS"
        options = optimoptions("patternsearch", "PollOrderAlgorithm", "Consecutive");
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
        [out,costs] = patternsearch(minfunc,x0,[],[],[],[],lb,ub,[],options);
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


%% main
string_array = ['holonotele4cm10x.tif', "holonotele4cm8x.tif",'HoloRBCnotele50x.tif',...
    'RBCnotele50x.tif','notele4cm.tif','holonotele.tif','hologramaNOTELE.tif'];
% string_array = ["holonotele4cm8x.tif"];
dim =    2; % 0 if you dont want to run minization, otherwise 2,4,8. 
cost =   1; % 0 - BIN; 1 - SD
% alg_array = ["GA+PS","GA","PS","FMC","FMM","FMU","FSO","PTS","SA","SGO"];
alg_array = ["GA+PS"];
for a = 1:size(alg_array,2)
    class_array = [];
    alg = alg_array(a);
    for i = 1:size(string_array,2)
        show = 1;
    %     lb = @(C)-.5/C;ub = @(C).5/C; % +- 0.5
        lb = @(C)-.5;ub = @(C).5; % +-C*50%
        if 1==1
            filename = string_array(i);
            [holo,M,N,X,Y] = holo_read(filename,0.5); %dont resize for any other images
            if show == 1
                figure,imagesc(holo),colormap(gray),title('Hologram'),daspect([1 1 1])
            end
            %%parameters
            if 1==1 % just to hide this stuff (remove visual clutter)
            grey_info = imfinfo(filename);
            bd = 2^grey_info.BitDepth;
            Lambda = 633*10^(-9);
            k = 2*pi/Lambda;
            dx = 6.9*10^(-6);
            dy = 6.9*10^(-6);
            erode =  0; % Gonna abandon this. 0 - off; 1 - on
            whichC = 0; % Gonna abandon this. 0 - avg(Cx,Cy); 1 - Cx; 2 - Cy
            end
            FT_holo = FT(holo);
            if show ==1 
                figure,imagesc(log(abs(FT_holo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) ;
            end 
            BW = threshold_FT(FT_holo, bd, M, N, erode, show);
            [plus_coor,m,n,p,q] = get_plus1(BW,show);
            p = plus_coor(1); q = plus_coor(2);
            off = 0;
            holoCompensate = filter_center_plus1(FT_holo,plus_coor-off,m-off,n-off,Lambda,X,Y,dx,dy,k, show);
            if show ==1 
                figure,imagesc(angle(holoCompensate)),colormap(gray),title('IFT Hologram'),daspect([1 1 1]) ;
            end
            bw = binarize_compensated_plus1(holoCompensate);
            if show == 1
                figure,imshow(bw),colormap(gray),title('Binarized Spherical Aberration'),daspect([1 1 1]) 
            end 
            [g,h] = get_g_and_h(bw,show);
            %%creating the new reference wave to eliminate the circular phase factors. 
            phi_spherical_C = @(C) pi/Lambda*((X-(g)+ 1).^2 + (Y-(h)+1).^2)*(dx^2)/(C);
            phi_spherical_CxCy = @(Cx,Cy) pi/Lambda*((X-(g)+ 1).^2 *dx^2 /(2*Cx)+ (Y-(h)+1).^2 *dy^2/(2*Cy));
            Cx = (M*dx)^2 / (Lambda*m); 
            Cy = (N*dy)^2 / (Lambda*n);
            phi_spherical = phi_spherical_C(Cy);
            % phi_spherical = phi_spherical_CxCy(Cx,Cy);
            phase_mask = exp((-1j)*phi_spherical);
            corrected_image = holoCompensate .* phase_mask;
            if show == 1
                figure; colormap gray; imagesc(angle(corrected_image));
                title('Auto Corrected image');axis square
            end
            if whichC == 0
                C = (Cx+Cy)/2;
            elseif whichC == 1
                C = Cx;
            elseif whichC == 2
                C = Cy;
            end
            r = Reconstruction;
            r.filename = string_array(i);
            r.C = C; r.Cx = Cx; r.Cy = Cy; r.seedCx = Cx; r.seedCy = Cy;
            r.holoCompensate = holoCompensate;
            r.Lambda = Lambda;
            r.X =X; r.Y = Y; r.g = g; r.h = h; r.dx = dx; r.dy = dy; r.m = m; r.n = n;
            r.p = p; r.q = q; r.k = k; r.bd = bd;
            r.corrected_image = corrected_image;
            r.filename = filename;
            r.holo = holo;

        end % kemper recon

        if dim == 1
            tic
            lb = lb(C); ub = ub(C);
            phi_spherical_C = @(C) (pi/(C*Lambda))*((X-(g)+ 1).^2 + (Y-(h)+1).^2)*(dx^2);
            rng default % For reproducibility
            init_pop_range_1d = @(C, lb, ub) [C+C*lb;C+C*ub];
            nvars = 1;        
            if     alg == "GA+PS"
                   hybridopts = optimoptions('patternsearch','Display','final');
                   options = optimoptions('ga', 'Display', 'final','HybridFcn', {@patternsearch,hybridopts},...
                        'InitialPopulationRange', init_pop_range_1d(C, lb, ub),...
                        'PopulationSize',15, ... % this speeds up convergence but not needed
                        'SelectionFcn','selectionremainder');
            elseif alg == "PS"
                    options = [];
            elseif  alg == "LSqNL"
                    options = optimoptions("lsqnonlin");
            end   %options

            if cost == 0
                minfunc = @(t)Raul_CF_noTele_BAR_1d(phi_spherical_C,t,holoCompensate,M,N);
            elseif cost == 1 
                minfunc = @(t)my_CF_noTele_BAR_1d(phi_spherical_C,t,holoCompensate,M,N);
            end  %select cost func

            if     alg == "GA+PS"
                    [out,costs] = ga(minfunc,nvars,[],[],[],[],[],[],[],options);
            elseif alg == "PS"
                    [out,costs] = patternsearch(minfunc,[C],[],[],[],[],[],[],[],options);
            elseif  alg == "LSqNL"
                    [out] = lsqnonlin(minfunc,C,lb,ub,options);
            end  %min

            toc
            if 1==1 %Reconstruct Hologram
                phi_spherical = phi_spherical_C(out);
                figtitle= ((alg)+' Min Reconstruction, C='+string(out));
            end
        end % not finished dont use

        if dim == 2
            tic
            lbx = lb(Cx); lby = lb(Cy); ubx = ub(Cx);uby = ub(Cy);
            phi_spherical_CxCy = @(Cx,Cy) pi/Lambda*((X-(g)+ 1).^2 *dx^2 /(2*Cx)+ (Y-(h)+1).^2 *dy^2/(2*Cy));
            rng default % For reproducibility
            x0 = [Cx, Cy];
            disp("seeds for Cx, Cy: "+string(x0))
            nvars = size(x0,2);
            init_pop_range_2d = @(Cx, Cy, lbx, ubx, lby, uby) [Cx+lbx*Cx, Cy+lby*Cy ; Cx+ubx*Cx, Cy+uby*Cy];
            if     alg == "GA+PS"
                    hybridopts = optimoptions('patternsearch','Display','final');
                    options = optimoptions(@ga,'HybridFcn', {@patternsearch,hybridopts},...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_2d(Cx, Cy, lbx, ubx, lby, uby), ...
                        'PopulationSize',15 ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder');
                        );
            elseif alg == "PS+GA"
                    hybridopts = optimoptions('ga','Display','final');
                    options = optimoptions(@patternsearch,'HybridFcn', {@ga,hybridopts},...
                        "PollOrderAlgorithm", "Consecutive"...
                        );
            elseif  alg == "GA"
                    options = optimoptions(@ga,...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_2d(Cx, Cy, lbx, ubx, lby, uby), ...
                        'PopulationSize',15, ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        'SelectionFcn','selectionremainder');
            elseif alg == "PS"
                    options = optimoptions("patternsearch", "PollOrderAlgorithm", "Consecutive");
            elseif alg == "FMC"
                    options = optimoptions("fmincon");
            elseif alg == "FMM"
                    options = optimoptions("fminimax");
            elseif alg == "FMU"
                    options = optimoptions("fminunc");
            elseif alg == "FSO"
                    options = optimoptions("fsolve");
            elseif  alg == "PTS"
                    options = optimoptions("paretosearch");
            elseif alg == "PSW"
                    options = optimoptions("particleswarm");
            elseif alg == "SA"
                    options = optimoptions("simulannealbnd");
            end  %options

            if cost == 0
                minfunc = @(t)Raul_CF_noTele_BAR_2d(phi_spherical_CxCy,t(1),t(2),holoCompensate,M,N);
            elseif cost == 1 
                minfunc = @(t)my_CF_noTele_BAR_2d(phi_spherical_CxCy,t(1),t(2),holoCompensate,M,N);
            end % select cost
            substr = alg.split('+');
            lb = [Cx+lbx*Cx; Cy+lby*Cy];
            ub = [Cx+ubx*Cx; Cy+uby*Cy];
            if     substr(1) == "GA"
                    [out,costs] = ga(minfunc,nvars,[],[],[],[],[],[],[],options);
            elseif alg == "PS"
                    [out,costs] = patternsearch(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMC"
                    [out,costs] = fmincon(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMM"
                    [out,costs] = fminimax(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMU"
                    [out,costs] = fminunc(minfunc,x0,options);
            elseif alg == "FSO"
                    [out,costs] = fsolve(minfunc,x0,options);
            elseif alg == "PTS"
                    [out,costs] = paretosearch(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PSW"
                   [out,costs] = particleswarm(minfunc,x0,lb,ub,options);
            elseif alg == "SA"
                   [out,costs] = simulannealbnd(minfunc,x0,lb,ub,options);
            elseif alg == "SGO"
                   [out,costs] = surrogateopt(minfunc,lb,ub);
            end  %min

            r.toc = toc;
            if 1==1
                %Reconstruct Hologram
                phi_spherical = phi_spherical_CxCy(out(1),out(2));
                figtitle = (alg)+' Min Reconstruction, Cx='+string(out(1))+', Cy='+string(out(2));
            end
        end
        
        if dim == 4
            tic
            lbx = lb(Cx); lby = lb(Cy); ubx = ub(Cx);uby = ub(Cy); 
            range_gh = 50;
            ubg = g+range_gh; lbg = g-range_gh; ubh = h+range_gh; lbh = h-range_gh;
            fun = @(Cx,Cy, g, h) pi/Lambda*((X-(g)+ 1).^2 *dx^2 /(2*Cx)+ (Y-(h)+1).^2 *dy^2/(2*Cy));
            rng default % For reproducibility
            x0 = [Cx, Cy, g, h];
            disp("seeds for Cx, Cy: "+string(x0))
            nvars = size(x0,2);
            init_pop_range_4d = @(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) [Cx+lbx*Cx, Cy+lby*Cy, lbg, lbg; ...
                                                                                    Cx+ubx*Cx, Cy+uby*Cy, ubg, ubh];
            if     alg == "GA+PS"
                    hybridopts = optimoptions('patternsearch','Display','final');
                    options = optimoptions(@ga,'HybridFcn', {@patternsearch,hybridopts},...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_4d(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) ...
                        ...'PopulationSize',15 ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder');
                        );
            elseif  alg == "GA"
                    options = optimoptions(@ga,...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_4d(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) ...
                        ...'PopulationSize',15, ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder'
                        );
            elseif alg == "PS"
                    options = optimoptions("patternsearch", "PollOrderAlgorithm", "Consecutive");
            elseif alg == "FMC"
                    options = optimoptions("fmincon");
            elseif alg == "FMM"
                    options = optimoptions("fminimax");
            elseif alg == "FMU"
                    options = optimoptions("fminunc");
            elseif alg == "FSO"
                    options = optimoptions("fsolve");
            elseif  alg == "PTS"
                    options = optimoptions("paretosearch");
            elseif alg == "PSW"
                    options = optimoptions("particleswarm");
            elseif alg == "SA"
                    options = optimoptions("simulannealbnd");
            end  %options

            if cost == 0
                minfunc = @(t)Raul_CF_noTele_BAR_4d(fun,t(1),t(2),t(3),t(4),holoCompensate,M,N);
            elseif cost == 1 
                minfunc = @(t)my_CF_noTele_BAR_4d(fun,t(1),t(2),t(3),t(4),holoCompensate,M,N);
            end % select cost
            substr = alg.split('+');
            lb = [Cx+lbx*Cx; Cy+lby*Cy; lbg; lbh];
            ub = [Cx+ubx*Cx; Cy+uby*Cy; ubg; ubh];
            if     substr(1) == "GA"
                    [out,costs] = ga(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PS"
                    [out,costs] = patternsearch(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMC"
                    [out,costs] = fmincon(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMM"
                    [out,costs] = fminimax(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMU"
                    [out,costs] = fminunc(minfunc,x0,options);
            elseif alg == "FSO"
                    [out,costs] = fsolve(minfunc,x0,options);
            elseif alg == "PTS"
                    [out,costs] = paretosearch(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PSW"
                   [out,costs] = particleswarm(minfunc,nvars,lb,ub,options);
            elseif alg == "SA"
                   [out,costs] = simulannealbnd(minfunc,x0,lb,ub,options);
            elseif alg == "SGO"
                   [out,costs] = surrogateopt(minfunc,lb,ub);
            end  %min

            r.toc = toc;
            if 1==1
                phi_spherical = fun(out(1),out(2),out(3),out(4));
                figtitle = (alg)+' Min Reconstruction, Cx='+string(out(1))+', Cy='+string(out(2))+...
                    ', g='+ string(out(3))+', h='+string(out(4));
            end%Reconstruct titles
        end
        
        if dim == 6
            tic
            g = 0; h=g;
            lbx = lb(Cx); lby = lb(Cy); ubx = ub(Cx);uby = ub(Cy); 
            range_gh = 50;
            ubg = g+range_gh; lbg = g-range_gh; ubh = h+range_gh; lbh = h-range_gh;
            range_mn = 10;
            ubm = m+range_mn; lbm = m-range_mn; ubn = n+range_mn; lbn = n-range_mn;
            range_pq  = 30;
            ubp = p+range_pq; lbp = p-range_pq; ubq = q+range_pq; lbq = q-range_pq;
            
            fun = @(Cx,Cy, g, h) pi/Lambda*((X-(g)+ 1).^2 *dx^2 /(2*Cx)+ (Y-(h)+1).^2 *dy^2/(2*Cy));
            rng default % For reproducibility
            x0 = [Cx, Cy, g, h, p, q]
            nvars = size(x0,2);
            init_pop_range_6d = @(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) [Cx+lbx*Cx, Cy+lby*Cy, lbg, lbh, lbp, lbq; ...
                                                                                    Cx+ubx*Cx, Cy+uby*Cy, ubg, ubh, ubp, ubq];
            if     alg == "GA+PS"
                    hybridopts = optimoptions('patternsearch','Display','final');
                    options = optimoptions(@ga,'HybridFcn', {@patternsearch,hybridopts},...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_6d(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) ...
                        ...'PopulationSize',15 ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder');
                        );
            elseif  alg == "GA"
                    options = optimoptions(@ga,...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_6d(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) ...
                        ...'PopulationSize',15, ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder'
                        );
            elseif alg == "PS"
                    options = optimoptions("patternsearch", "PollOrderAlgorithm", "Consecutive");
            elseif alg == "FMC"
                    options = optimoptions("fmincon");
            elseif alg == "FMM"
                    options = optimoptions("fminimax");
            elseif alg == "FMU"
                    options = optimoptions("fminunc");
            elseif alg == "FSO"
                    options = optimoptions("fsolve");
            elseif  alg == "PTS"
                    options = optimoptions("paretosearch");
            elseif alg == "PSW"
                    options = optimoptions("particleswarm");
            elseif alg == "SA"
                    options = optimoptions("simulannealbnd");
            end  %options

            if cost == 0
                minfunc = @(t)Raul_CF_noTele_BAR_6d(fun,t(1),t(2),t(3),t(4),t(5),t(6),m,n,FT_holo,M,N,Lambda,X,Y,dx,dy,k, show);
            elseif cost == 1 
                minfunc = @(t)my_CF_noTele_BAR_6d(fun,t(1),t(2),t(3),t(4),t(5),t(6),m,n,FT_holo,M,N,Lambda,X,Y,dx,dy,k, show);
            end % select cost
            substr = alg.split('+');
            lb = [Cx+lbx*Cx; Cy+lby*Cy; lbg; lbh; lbp; lbq];
            ub = [Cx+ubx*Cx; Cy+uby*Cy; ubg; ubh; ubp; ubq];
            if     substr(1) == "GA"
                    [out,costs] = ga(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PS"
                    [out,costs] = patternsearch(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMC"
                    [out,costs] = fmincon(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMM"
                    [out,costs] = fminimax(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMU"
                    [out,costs] = fminunc(minfunc,x0,options);
            elseif alg == "FSO"
                    [out,costs] = fsolve(minfunc,x0,options);
            elseif alg == "PTS"
                    [out,costs] = paretosearch(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PSW"
                   [out,costs] = particleswarm(minfunc,nvars,lb,ub,options);
            elseif alg == "SA"
                   [out,costs] = simulannealbnd(minfunc,x0,lb,ub,options);
            elseif alg == "SGO"
                   [out,costs] = surrogateopt(minfunc,lb,ub);
            end  %min

            r.toc = toc;
            holoCompensate = filter_center_plus1(FT_holo,[out(5),out(6)],m,n,Lambda,X,Y,dx,dy,k, show);
            phi_spherical = fun(out(1),out(2),out(3),out(4));
                
        end

        if dim == 8
            tic
            lbx = lb(Cx); lby = lb(Cy); ubx = ub(Cx);uby = ub(Cy); 
            range_gh = 50;
            ubg = g+range_gh; lbg = g-range_gh; ubh = h+range_gh; lbh = h-range_gh;
            range_mn = 10;
            ubm = m+range_mn; lbm = m-range_mn; ubn = n+range_mn; lbn = n-range_mn;
            range_pq  = 50;
            ubp = p+range_pq; lbp = p-range_pq; ubq = q+range_pq; lbq = q-range_pq;
            
            fun = @(Cx,Cy, g, h) pi/Lambda*((X-(g)+ 1).^2 *dx^2 /(2*Cx)+ (Y-(h)+1).^2 *dy^2/(2*Cy));
            rng default % For reproducibility
            x0 = [Cx, Cy, g, h, m, n, p, q]
            nvars = size(x0,2);
            init_pop_range_8d = @(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) [Cx+lbx*Cx, Cy+lby*Cy, lbg, lbh, lbm, lbn, lbp, lbq; ...
                                                                                    Cx+ubx*Cx, Cy+uby*Cy, ubg, ubh, ubm, ubn, ubp, ubq];
            if     alg == "GA+PS"
                    hybridopts = optimoptions('patternsearch','Display','final');
                    options = optimoptions(@ga,'HybridFcn', {@patternsearch,hybridopts},...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_8d(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) ...
                        ...'PopulationSize',15 ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder');
                        );
            elseif  alg == "GA"
                    options = optimoptions(@ga,...
                        'InitialPopulationMatrix', x0,...
                        'InitialPopulationRange',  init_pop_range_8d(Cx, Cy, lbx, ubx, lby, uby, ubg, ubh, lbg, lbh) ...
                        ...'PopulationSize',15, ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                        ...'SelectionFcn','selectionremainder'
                        );
            elseif alg == "PS"
                    options = optimoptions("patternsearch", "PollOrderAlgorithm", "Consecutive");
            elseif alg == "FMC"
                    options = optimoptions("fmincon");
            elseif alg == "FMM"
                    options = optimoptions("fminimax");
            elseif alg == "FMU"
                    options = optimoptions("fminunc");
            elseif alg == "FSO"
                    options = optimoptions("fsolve");
            elseif  alg == "PTS"
                    options = optimoptions("paretosearch");
            elseif alg == "PSW"
                    options = optimoptions("particleswarm");
            elseif alg == "SA"
                    options = optimoptions("simulannealbnd");
            end  %options

            if cost == 0
                minfunc = @(t)Raul_CF_noTele_BAR_8d(fun,t(1),t(2),t(3),t(4),t(5),t(6),t(7),t(8),FT_holo,M,N,Lambda,X,Y,dx,dy,k, show);
            elseif cost == 1 
                minfunc = @(t)my_CF_noTele_BAR_8d(fun,t(1),t(2),t(3),t(4),t(5),t(6),t(7),t(8),FT_holo,M,N,Lambda,X,Y,dx,dy,k, show);
            end % select cost
            substr = alg.split('+');
            lb = [Cx+lbx*Cx; Cy+lby*Cy; lbg; lbh; lbm; lbn; lbp; lbq];
            ub = [Cx+ubx*Cx; Cy+uby*Cy; ubg; ubh; ubm; ubn; ubp; ubq];
            if     substr(1) == "GA"
                    [out,costs] = ga(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PS"
                    [out,costs] = patternsearch(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMC"
                    [out,costs] = fmincon(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMM"
                    [out,costs] = fminimax(minfunc,x0,[],[],[],[],lb,ub,[],options);
            elseif alg == "FMU"
                    [out,costs] = fminunc(minfunc,x0,options);
            elseif alg == "FSO"
                    [out,costs] = fsolve(minfunc,x0,options);
            elseif alg == "PTS"
                    [out,costs] = paretosearch(minfunc,nvars,[],[],[],[],lb,ub,[],options);
            elseif alg == "PSW"
                   [out,costs] = particleswarm(minfunc,nvars,lb,ub,options);
            elseif alg == "SA"
                   [out,costs] = simulannealbnd(minfunc,x0,lb,ub,options);
            elseif alg == "SGO"
                   [out,costs] = surrogateopt(minfunc,lb,ub);
            end  %min

            r.toc = toc;
            holoCompensate = filter_center_plus1(FT_holo,[out(7),out(8)],out(5),out(6),Lambda,X,Y,dx,dy,k, 0);
            phi_spherical = fun(out(1),out(2),out(3),out(4));
                
        end
        
        if 1==1%Reconstruct Hologram
            show = 1;
            phase_mask = exp((-1j)*phi_spherical);
            corrected_image = holoCompensate .* phase_mask;
            if show == 1
                figure; colormap gray; mesh((angle(corrected_image)));axis square
                 title(figtitle);
            end
            r.Cx = out(1);
            r.Cy = out(2);
            if dim >= 4
                r.g = out(3);
                r.h = out(4);
                if dim == 6
                    r.p = out(5);
                    r.q = out(6);
                elseif dim == 8
                    r.m = out(5);
                    r.n = out(6);
                    r.p = out(7);
                    r.q = out(8);
                end
            end
            r.holoCompensate = holoCompensate;
            r.corrected_image = corrected_image;
            r.phase_mask = phase_mask;
            r.phi_spherical = phi_spherical;
            class_array = [r, class_array];
        end
    end
%%Save
if cost == 0 
    savefilename = "kemper_recon"+string(dim)+"d_1209_BIN_"+string(alg)+".mat";
elseif cost == 1
    savefilename = "kemper_recon"+string(dim)+"d_1209_TSD_"+string(alg)+".mat";
end
save(savefilename, 'class_array')
end
%% Functions
function [J] = my_CF_noTele_BAR_6d(fun,Cx,Cy,g,h,p,q,m,n,FT_holo,M,N, Lambda,X,Y,dx,dy,k, show)
    holoCompensate = filter_center_plus1(FT_holo,[p,q],m,n,Lambda,X,Y,dx,dy,k, 0);
    phi_spherical = fun(Cx,Cy,g,h);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    J = std(phase,[],'all');
end
function [J] = my_CF_noTele_BAR_8d(fun,Cx,Cy,g,h,m,n,p,q,FT_holo,M,N, Lambda,X,Y,dx,dy,k, show)
    holoCompensate = filter_center_plus1(FT_holo,[p,q],m,n,Lambda,X,Y,dx,dy,k, 0);
    phi_spherical = fun(Cx,Cy,g,h);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    J = std(phase,[],'all');
end
function [J] = Raul_CF_noTele_BAR_4d(fun,Cx,Cy,g,h,holoCompensate,M,N)
    phi_spherical = fun(Cx,Cy,g,h);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    ib = imbinarize(phase, 0.5);
    J = M*N - sum(ib(:));
end
function [J] = my_CF_noTele_BAR_4d(fun,Cx,Cy,g,h,holoCompensate,M,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: CF_noTele_BAR0531                                                     %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi_spherical = fun(Cx,Cy,g,h);
%     phi_spherical = fun(Cx,Cy);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    J = std(phase,[],'all');
end
function [J] = Raul_CF_noTele_BAR_3d(fun,C,g,h,holoCompensate,M,N)
    phi_spherical = fun(C,g,h);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    ib = imbinarize(phase, 0.5);
    J = M*N - sum(ib(:));
end
function [J] = my_CF_noTele_BAR_3d(fun,C,g,h,holoCompensate,M,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: CF_noTele_BAR0531                                                     %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi_spherical = fun(C,g,h);
%     phi_spherical = fun(Cx,Cy);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    J = std(phase,[],'all');
end
function corrected_image = reconstruct_3D(holoCompensate,X,Y,C,g,h,plot_title,fun, show)
    phi_spherical = fun(C,g,h);
    phi_total = phi_spherical;
    phase_mask = exp((-1j)*phi_total);
    %Reconstruct Hologram
    corrected_image = holoCompensate .* phase_mask;
    if show == 1
        figure; colormap gray; imagesc(angle(corrected_image));title(plot_title);axis square
    end
end
function [J] = Raul_CF_noTele_BAR_2d(fun,Cx,Cy,holoCompensate,M,N)
    phi_spherical = fun(Cx,Cy);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    ib = imbinarize(phase, 0.5);
    J = M*N - sum(ib(:));
end
function [J] = my_CF_noTele_BAR_2d(fun,Cx,Cy,holoCompensate,M,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: CF_noTele_BAR0531                                                     %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi_spherical = fun(Cx,Cy);
%     phi_spherical = fun(Cx,Cy);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    J = std(phase,[],'all');
end







