%{
Title: no_tele_main.m

Description: This matlab script provides phase compensation for DHM holograms recorded in a non-telecentric configuration. 
The code was originally developed by Brian Bogue-Jimenez, with assistance from Prof. Ana Doblas and Dr. Raúl Castañeda.
This matlab version of the code has been fully automated, requiring only that the user input the pixel size 
in each dimension (dx and dy) and the illumination wavelength (Lambda).

Dependencies: The code uses functions from the 'funs.m' script.

Date: January 6, 2023
Last update: March 17, 2023.

Authors: Brian Bogue-Jimenez, Carlos Trujillo, and Ana Doblas
%}

%% clear memory 

clc % clean
close all% close all windows
clear all% clear of memory all variable

input_argument2 = 5;

%addpath('directory_containing_my_function_script')
output2 = funs(input_argument2);


%% The main code starts here

% Different image files to process (no-tele holograms). Sample input holograms located in 'data/'
% ["4cm_20x_bigcakes.tiff", "-4cm_20x_star.tiff", "4cm_20x_usaf.tiff", "RBCnotele50x.tiff"]

% Loading image file (hologram) to process
user_input = '-4cm_20x_star_small.tiff';
filename = ['data/', user_input];
disp(['Non-telecentric DHM hologram: ', filename]);



%% sadsad

vargin = 1; % Scalling factor of the input images
[holo, M, N, X, Y] = holo_read(filename, vargin);

figure; imshow(holo, 'InitialMagnification', 'fit'); title('Hologram');
daspect([1 1 1]); % equal aspect ratio
show;

% Variables and flags for hologram reconstruction (Default variables, change accordingly)
% Lambda = 53210^(-9)
Lambda = 63310^(-9);
k = 2*pi/Lambda;
dx = 6.910^(-6);
dy = 6.9*10^(-6);

disp('Phase compensation starts...');



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
function [J] = Raul_CF_noTele_BAR_1d(fun,seed_cur,holoCompensate,M,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: CF_noTele_BAR0531                                                     %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi_spherical = fun(seed_cur);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    ib = imbinarize(phase, 0.5);
    J = M*N - sum(ib(:));
end    
function [J] = my_CF_noTele_BAR_1d(fun,Cx,holoCompensate,M,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: CF_noTele_BAR0531                                                     %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi_spherical = fun(Cx);
%     phi_spherical = fun(Cx,Cy);
    phase_mask = exp((-1j)*phi_spherical);
    corrected_image = holoCompensate .* phase_mask;
    phase = angle(corrected_image);
    phase = phase + pi; 
    J = std(phase,[],'all');
    if Cx == 0
        J = 0.5;
    end
end
function [g,h] = get_g_and_h(bw, show) 

    cc = bwconncomp(bw,4);

    numPixels = cellfun(@numel,cc.PixelIdxList);
    num_of_terms = 3;
    idx_list = zeros(num_of_terms,1);


    for i = 1:num_of_terms
        [biggest,idx] = max(numPixels);
        idx_list(i) = idx;
        numPixels(idx) = 0;
    end

    for i = 1:length(cc.PixelIdxList)
        if ~(i == idx_list(1) || i == idx_list(2) || i == idx_list(3))
            bw(cc.PixelIdxList{i}) = 0;
        end
    end

    terms = regionprops(bw,'all');
    [best_attr,best_term_idx] = min([terms.Eccentricity]);
    best_term = terms(best_term_idx).BoundingBox;
    block_center = [best_term(1)+best_term(3)/2, best_term(2)+best_term(4)/2];
    [M,N] = size(bw);
    true_center = [M/2, N/2];

    g_and_h = abs(block_center - true_center);
    if show == 1
        figure, imshow(bw);
        axis on;
        hold on;
        plot(true_center(1), true_center(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
        plot(block_center(1), block_center(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
        line([true_center(1),block_center(1)],[true_center(2),block_center(2)],'Color','red','LineStyle','--')
        rectangle('Position', terms(best_term_idx).BoundingBox, 'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');
    end
    % values for g and h
    g = g_and_h(1);
    h = g_and_h(2);
    disp("G: "+string(g)+" H: "+string(h));
end
function bw = binarize_compensated_plus1(holoCompensate) 
    bw = imbinarize(real(holoCompensate)); %binarize the image
end
function holoCompensate = filter_center_plus1(FT_holo,plus_coor,m,n,Lambda,X,Y,dx,dy,k, show) 
    [M,N] = size(FT_holo);
    Filter = zeros(M,N);
    warning('off','all')
    Filter((plus_coor(2) - n) :(plus_coor(2)+ n) ,(plus_coor(1) - m) :(plus_coor(1)+ m))=1;
    FT_FilteringH = FT_holo.*Filter;
    if show == 1
        figure,imagesc(log((abs(FT_FilteringH).^2))),colormap(gray),title('FT Hologram filter'),daspect([1 1 1]) 
    end
    ThetaXM = asin((M/2 - plus_coor(1))*Lambda/(M * dx));
    ThetaYM = asin((N/2 - plus_coor(2))*Lambda/(N * dy));
    Reference = exp(1i*k*(sin(ThetaXM)*X*dx + sin(ThetaYM)*Y*dy));

    holo_filtered = fftshift(ifft2(fftshift(FT_FilteringH)));

    % figure,imagesc(log(abs(holo_filtered)).^2),colormap(gray),title('Amplitude filtered'),daspect([1 1 1]) 
%     figure,imagesc(angle(holo_filtered)),colormap(gray),title('Hologram filtered'),daspect([1 1 1]) 

    holoCompensate = holo_filtered.*Reference;
end
function BW = threshold_FT(FT_holo, bd, M, N, erode, show)
    if nargin == 4
        show = 1;set
    end
    I = sqrt(abs(FT_holo));
    px = 30; I(M/2-px:M/2+px, N/2-px:N/2+px) = 0;
    [counts,x] = imhist(I,16);
    T = otsuthresh(counts);
    if erode == 1
        BW = imbinarize(erodedI, T*bd);
    elseif erode == 0
        BW = imbinarize(I, T*bd);
    end
    if show == 1
        figure; imagesc(BW);axis square; colormap gray;
    end
end
function [plus_coor,m,n,p,q] = get_plus1(bw, show) 
    if nargin == 1
        show = 1;
    end
    cc = bwconncomp(bw,4);
    numPixels = cellfun(@numel,cc.PixelIdxList);
    idx_list = zeros(2,1);
    for i = 1:2
        [biggest,idx] = max(numPixels);
        idx_list(i) = idx;
        numPixels(idx) = 0;
    end
    for i = 1:length(cc.PixelIdxList)
        if ~(i == idx_list(1) || i == idx_list(2) )
            bw(cc.PixelIdxList{i}) = 0;
        end
    end
    terms = regionprops(bw);
    plus1 = terms(1).BoundingBox;
    plus_coor = [plus1(1)+plus1(3)/2, plus1(2)+plus1(4)/2]; 
    [M,N] = size(bw);
    dc_coor = [M/2, N/2];
    p_and_q = abs(plus_coor - dc_coor); 
    if show == 1
        figure, imshow(bw);
        axis on;
        hold on;
        plot(plus_coor(1), plus_coor(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
        plot(dc_coor(1), dc_coor(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);

        line([plus_coor(1),dc_coor(1)],[plus_coor(2),dc_coor(2)],'Color','red','LineStyle','--')
        for i = 1:numel(terms)
            rectangle('Position', terms(i).BoundingBox, ...
            'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');
        end
    end
    box_size = terms(1).BoundingBox;
    % values for p,q,m and n (min and kemper paper)
    m = box_size(3)/2 ;
    n = box_size(4)/2 ;
    p = p_and_q(1);
    q = p_and_q(2);
    disp("P: "+string(p)+" Q: "+string(q));
    disp("M: "+string(m)+" N: "+string(n));
end
function ft = FT(holo)
    ft = fftshift(fft2(fftshift(holo)));
end
function [holo,M,N,m,n] = holo_read(filename,vargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: holo_read                                                             %
%                                                                              %
% The function is implemented to read a hologram, the hologram is cut to get   %
% square dimension, the square has the dimension of the higher side of the     %
% digital camera used to record the hologram, and the cut is making from the   %
% center of the hologram.                                                      %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    holo = double(imread(filename));
    holo = holo(:,:,1);
    if nargin == 2
        scale = vargin;
    else
        scale = 1;
    end
    holo = imresize(holo,scale,'nearest');
    [M,N] = size(holo);
        if (M > N)
            cut = (M - N)/2;
            holo = holo(cut:(M-cut-1):N,1:N);
        elseif (M < N)
            cut = (N - M)/2;
            holo = holo(1:M,cut:(N-cut)-1);
        else
            holo = holo;
        end
    [M,N] = size(holo);    
    [m,n] = meshgrid(-M/2+1:M/2,-N/2+1:N/2);
end