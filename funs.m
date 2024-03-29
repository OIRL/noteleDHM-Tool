classdef funs
    methods(Static)

          function [holo,M,N,X,Y] = holo_read(filename,vargin)
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

%             holo = imresize(holo,scale,'nearest');
%             [M,N] = size(holo);
%             if (M > N)
%                 cut = (M - N)/2;
%                 holo = holo(cut:(M-cut-1):N,1:N);
%             elseif (M < N)
%                 cut = (N - M)/2;
%                 holo = holo(1:M,cut:(N-cut)-1);
%             else
%                 holo = holo;
%             end
            [N,M] = size(holo);    
            [X,Y] = meshgrid(-M/2+1:M/2,-N/2+1:N/2); %The order of the variables must be switched (MATLAB)

          end

          function ft = FT(holo)
            ft = fftshift(fft2(fftshift(holo)));
          end

          function BW = threshold_FT(FT_holo, M, N)
            bd = 256;
            I = sqrt(abs(FT_holo));
            px = 30; I(M/2-px:M/2+px, N/2-px:N/2+px) = 0;
            [counts,x] = imhist(I,16);
            T = otsuthresh(counts);

            BW = imbinarize(I, T*bd*1.4);

            %figure; imagesc(BW);axis square; colormap gray;

          end
          
          function [plus_coor,m,n,p,q] = get_plus1(bw, M, N) 
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
            %[N,M] = size(bw);
            dc_coor = [M/2, N/2];
            p_and_q = abs(plus_coor - dc_coor); 

                figure, imshow(bw);
                axis on;
                hold on;
                plot(plus_coor(1), plus_coor(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
                plot(dc_coor(1), dc_coor(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);

                line([plus_coor(1),dc_coor(1)],[plus_coor(2),dc_coor(2)],'Color','red','LineStyle','--')
                for i = 1:numel(terms)
                    rectangle('Position', terms(i).BoundingBox, ...
                'LineWidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');

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
          
          function holoCompensate = filter_center_plus1(FT_holo,plus_coor,m,n,Lambda,X,Y, M,N,dx,dy,k) 
            %[M,N] = size(FT_holo);
            Filter = zeros(N,M);
            warning('off','all')
            Filter((plus_coor(2) - n) :(plus_coor(2)+ n) ,(plus_coor(1) - m) :(plus_coor(1)+ m))=1;
            FT_FilteringH = FT_holo.*Filter;

            %figure,imagesc(log((abs(FT_FilteringH).^2))),colormap(gray),title('FT Hologram filter'),daspect([1 1 1]) 

            ThetaXM = asin((M/2 - plus_coor(1))*Lambda/(M * dx));
            ThetaYM = asin((N/2 - plus_coor(2))*Lambda/(N * dy));
            Reference = exp(1i*k*(sin(ThetaXM)*X*dx + sin(ThetaYM)*Y*dy));

            holo_filtered = fftshift(ifft2(fftshift(FT_FilteringH)));

            % figure,imagesc(log(abs(holo_filtered)).^2),colormap(gray),title('Amplitude filtered'),daspect([1 1 1]) 
            % figure,imagesc(angle(holo_filtered)),colormap(gray),title('Hologram filtered'),daspect([1 1 1]) 

            holoCompensate = holo_filtered.*Reference;
          end

          function bw = binarize_compensated_plus1(holoCompensate) 
            bw = imbinarize(real(holoCompensate)); %binarize the image
          end

          function [g,h] = get_g_and_h(bw, M, N) 

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
            %[M,N] = size(bw);
            true_center = [M/2, N/2];

            g_and_h = (block_center - true_center)/1;

            figure, imshow(bw);
            axis on;
            hold on;
            plot(true_center(1), true_center(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
            plot(block_center(1), block_center(2), 'b*', 'MarkerSize', 10, 'LineWidth', 2);
            line([true_center(1),block_center(1)],[true_center(2),block_center(2)],'Color','yellow','LineStyle','--')
            rectangle('Position', terms(best_term_idx).BoundingBox, 'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');

            % values for g and h
            g = g_and_h(1);
            h = g_and_h(2);
            disp("G: "+string(g)+" H: "+string(h));
          end

          function [J] = bin_CF_noTele_BAR_1d(fun,seed_cur,holoCompensate,M,N,sign)
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
            if sign
                phase_mask = exp((1j)*phi_spherical);
            else
                phase_mask = exp((-1j)*phi_spherical);
            end
            corrected_image = holoCompensate .* phase_mask;
            phase = angle(corrected_image);
            phase = phase + pi; 
            ib = imbinarize(phase, 0.5);
            J = M*N - sum(ib(:));
          end    
          
          function [J] = std_CF_noTele_BAR_1d(fun,Cx,holoCompensate,M,N, sign)
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
            if sign
                phase_mask = exp((1j)*phi_spherical);
            else
                phase_mask = exp((-1j)*phi_spherical);
            end
            corrected_image = holoCompensate .* phase_mask;
            phase = angle(corrected_image);
            phase = phase + pi; 
            J = std(phase,[],'all');
            if Cx == 0
                J = 0.5;
            end
          end

          function [corrected_image] = automatic_method(holo, M, N, X, Y, Lambda, dx, dy, algo, cost)

            k = 2*pi/Lambda;
            
            %Let's go to the spatial frequency domain
            FT_holo = funs.FT(holo);
            %figure,imagesc(log(abs(FT_holo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) ;

            %Let's threshold that FT
            BW = funs.threshold_FT(FT_holo, M, N);

            %Get the +1 D.O. term region and coordinates
            %tic
            [plus_coor,m,n,p,q] = funs.get_plus1(BW, M, N);
            %toc

            %Compensating the tilting angle first
            holoCompensate = funs.filter_center_plus1(FT_holo,plus_coor,m,n,Lambda,X,Y,M,N,dx,dy,k);
            figure,imagesc(angle(holoCompensate)),colormap(gray),title('Tilting angle compensated hologram'),daspect([1 1 1]);
    
            %Binarized Spherical Aberration
            bw = funs.binarize_compensated_plus1(holoCompensate);
            %figure,imshow(bw),colormap(gray),title('Binarized remaining Spherical Aberration'),daspect([1 1 1]) 

            %Get the center of the remaining spherical phase factor for the 2nd compensation
            [g,h] = funs.get_g_and_h(bw, M, N);

            %%Let's create the new reference wave to eliminate the circular phase factors
            phi_spherical_C = @(C) (pi/Lambda)*( ((X-g).^2 * dx^2 / C) + ((Y-h).^2 * dy^2 / C) );
    
            Cx = (M*dx)^2 / (Lambda*m); 
            Cy = (N*dy)^2 / (Lambda*n);
            cur = (Cx + Cy)/2;

            %Let's select the sign of the compensating spherical phase factor
            sign = true;
    
            %Let's built the spherical phase factor for compensation
            phi_spherical = phi_spherical_C(cur);

            if sign
                phase_mask = exp((1j)*phi_spherical);
            else
                phase_mask = exp((-1j)*phi_spherical);
            end

%             figure; colormap gray; imagesc(angle(phase_mask));
%             title('Compensating spherical phase factor')
% 
%             figure; colormap gray; imagesc(angle(holoCompensate));
%             title('holoCompensate to compensate')

            corrected_image = holoCompensate .* phase_mask;
            figure; colormap gray; imagesc(angle(corrected_image));
            title('Non-optimized compensated image');
            
            % Up to this point the phase compensation for no telecentric DHM holograms is finished according to Kemper
    
            %Set the default random number generator for reproducibility
            rng(0);
    
            %Different available optimization methods
            alg_array = ["FMC","FMU","FSO","SA","PTS","GA","PS","GA+PS"];
            alg = alg_array(algo);

            % Two available cost functions
            cost_fun = {'BIN cost function', 'STD cost function'};
            % cost = 0 for BIN, 1 for SD (See documentation for further details)
            disp(['Selected cost function: ', cost_fun{cost+1}])

            if cost == 0
                minfunc = @(t)funs.bin_CF_noTele_BAR_1d(phi_spherical_C,t,holoCompensate,M,N,sign);
            elseif cost == 1 
                minfunc = @(t)funs.std_CF_noTele_BAR_1d(phi_spherical_C,t,holoCompensate,M,N,sign);
            end  %select cost func

            %Determination (minimization) of the optimal parameter (C -curvature) for the accurate phase compensation of no tele DHM holograms.
    
            %Define the lower and upper bounds of the initial population range (Warning: Modifying these settings may cause unexpected behavior. Proceed with caution)
            lb = @(C)-.5;ub = @(C).5; % +-C*50%
            lb = lb(cur); ub = ub(cur);
            x0 = cur;
            nvars = 1;
    
            disp(['cur: ', num2str(cur)]);

            % Options for the Optimization functions
            if alg == "GA"
                disp(['Running the genetic algorithm with the ', cost_fun{cost+1}])
                init_pop_range_1d = @(C, lb, ub) [C+C*lb;C+C*ub];
                options = optimoptions(@ga,...
                'InitialPopulationMatrix', x0,...
                'InitialPopulationRange',  init_pop_range_1d(x0, lb, ub), ...
                'PopulationSize',15, ... % this speeds up convergence but not needed, default is 50 when nvars<=5
                'SelectionFcn','selectionremainder');
            elseif alg == "GA+PS"
                disp(['Running the hybrid (GA + PS) algorithm with the ', cost_fun{cost+1}])
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
          
          end

          %{
            ##################################################################
            ################Functions for manual determination################
            ##################################################################
          %}

          function [Xcenter, Ycenter, holo_filter, ROI_array] = spatialFilterinCNT(inp, M, N)

            fft_holo = fft2(single(inp)); % FFT of hologram
            fft_holo = fftshift(fft_holo);
            fft_holo_image = 20*log10(abs(fft_holo)); % logarithmic scale FFT

            minVal = min(min(abs(fft_holo_image)));
            maxVal = max(max(abs(fft_holo_image)));
            fft_holo_image = uint8((fft_holo_image - minVal) * 255 / (maxVal - minVal)); % conversion of data to uint8

            % select ROI interactively
            imshow(fft_holo_image); title("Select the +1 ROI")
            r1 = drawrectangle('Label','+1 D.O. ROI','Color',[1 0 0]);
            ROI = r1.Position;

            x1_ROI = fix(ROI(2));
            y1_ROI = fix(ROI(1));
            x2_ROI = fix(ROI(2) + ROI(4));
            y2_ROI = fix(ROI(1) + ROI(3));
            ROI_array = [x1_ROI, y1_ROI, x2_ROI, y2_ROI];

            % computing the center of the rectangle mask
            Ycenter = x1_ROI + (x2_ROI - x1_ROI)/2;
            Xcenter = y1_ROI + (y2_ROI - y1_ROI)/2;

            holo_filter = zeros(N, M);

            holo_filter(x1_ROI:x2_ROI, y1_ROI:y2_ROI, :) = 1;
           
            holo_filter = holo_filter .* fft_holo;
            holo_filter = ifftshift(holo_filter);
            holo_filter = ifft2(holo_filter);

          end

          function [phaseCompensate, phi_spherical, summ] = TSM(comp_phase, curTemp, gTemp, hTemp, wavelength, X, Y, dx, dy, sign)
            phi_spherical = ((X - gTemp).^2 * dx^2 / curTemp) + ((Y - hTemp).^2 * dy^2 / curTemp);
            phi_spherical = pi * phi_spherical / wavelength;

            if (sign)
                phi_spherical = exp(1i * phi_spherical);
            else
                phi_spherical = exp(-1i * phi_spherical);
            end
    
            phaseCompensate = comp_phase .* phi_spherical;
            phaseCompensate = angle(phaseCompensate);
    
            minVal = min(phaseCompensate(:));
            maxVal = max(phaseCompensate(:));
            phase_sca = (phaseCompensate - minVal) / (maxVal - minVal);
            
            %STD:
            summ = 1 - std(phase_sca(:));
            
            %TSM:
            %binary_phase = (phase_sca > 0.2);

            % Applying the summation and thresholding metric
            %summ = sum(binary_phase(:));
            
          end

          function [g_out, h_out, cur_out, sum_max] = brute_search(comp_phase, arrayCurvature, arrayXcenter, arrayYcenter, wavelength, X, Y, dx, dy, sign, vis)  
            sum_max = 0;
            cont = 0;
            for curTemp = arrayCurvature
                %disp('arrayCurvature');
                for gTemp = arrayXcenter
                    for hTemp = arrayYcenter
                        cont = cont + 1;
            
                        [phaseCompensate, phi_spherical, sum_val] = funs.TSM(comp_phase, curTemp, gTemp, hTemp, wavelength, X, Y, dx, dy, sign);
            
                        if (sum_val > sum_max)
                            g_out = gTemp;
                            h_out = hTemp;
                            cur_out = curTemp;
                            sum_max = sum_val;
                        end

                        if vis
                            % Create a figure with three subplots
                            figure;

                                % Plot the first image in the first subplot
                                subplot(1, 3, 1);
                                imagesc(angle(phi_spherical));
                                title('phi_spherical');axis square

                                % Plot the second image in the second subplot
                                subplot(1, 3, 2);
                                imagesc(angle(comp_phase));
                                title('comp_phase');axis square

                                % Plot the third image in the third subplot
                                subplot(1, 3, 3);
                                imagesc(phaseCompensate);
                                title('phaseCompensate');axis square

                                % Show the figure
                                sgtitle('Subplots');
                        end

                    end
                end
            end

          end

          function [phaseCompensate] = fast_CNT(holo, Lambda, M, N, X, Y, dx, dy)

            % Function for rapid compensation of phase maps of image plane off-axis DHM, operating in non-telecentric regimen
            % Inputs:
            % inp - The input intensity (captured) hologram
            % wavelength - Wavelength of the illumination source to register the DHM hologram
            % dx, dy - Pixel dimensions of the camera sensor used for recording the hologram
            
            k = 2*pi/Lambda;
            
            % The spatial filtering process is executed
            disp('Spatial filtering process started.....');

            [Xcenter, Ycenter, holo_filter, ROI_array] = funs.spatialFilterinCNT(holo, M, N);

            disp('Spatial filtering process finished.');

            %figure, imagesc(abs(holo_filter).^2), colormap gray, title('holo filter')

            % Fourier transform of the hologram filtered
            ft_holo = funs.FT(holo_filter);

            %figure; colormap gray; imagesc(abs(ft_holo).^2);
            %title('FT Filtered holo');axis square

            % reference wave for the first compensation (global linear compensation)
            ThetaXM = asin((M/2 - Xcenter) * Lambda / (M * dx));
            ThetaYM = asin((N/2 - Ycenter) * Lambda / (N * dy));
            reference = exp(1j * k * (sin(ThetaXM) * X * dx + sin(ThetaYM) * Y * dy));

            % First compensation (tilting angle compensation)
            comp_phase = holo_filter .* reference;
            phase_c = angle(comp_phase);

            % show the first compensation
            minVal = min(phase_c(:));
            maxVal = max(phase_c(:));
            phase_normalized = (phase_c - minVal) / (maxVal - minVal);
            binary_phase = (phase_normalized > 0.5);
            %figure; colormap gray; imagesc(binary_phase);
            %title('binary phase');axis square

            % creating the new reference wave to eliminate the circular phase factors (second phase compensation)
            m = abs(ROI_array(3) - ROI_array(1));
            n = abs(ROI_array(4) - ROI_array(2));
            Cx = (M * dx)^2 / (Lambda * m);
            Cy = (N * dy)^2 / (Lambda * n);
            cur = (Cx + Cy)/2;

            % display the phase_normalized image
            imshow(phase_normalized); title("Click the center of the spherical phase factor")

            % use ginput to get the coordinates of a click
            [x, y] = ginput(1);

            % print out the coordinates
            fprintf('The coordinates of the center of the spherical phase factor are: (%f, %f)\n', x, y);

            p = x;
            q = y;

            g = ((M/2) - fix(p))/2;
            h = ((N/2) - fix(q))/2;

            % Let's find the sign of the spherical phase factor
            %(3D Grid search to assess this value)
            s = 2;
            perc = 0.1;
            arrayCurvature = [(cur - (cur*perc)) cur (cur + (cur*perc))];
            arrayXcenter = [(g - 2*s) (g - s) g (g + s) (g + 2*s)];
            arrayYcenter = [(h - 2*s) (h - s) h (h + s) (h + 2*s)];

            sign = true;
            [~, ~, ~, sum_max_True] = funs.brute_search(comp_phase, arrayCurvature, arrayXcenter, arrayYcenter, Lambda, X, Y, dx, dy, sign, false);

            sign = false;
            [~, ~, ~, sum_max_False] = funs.brute_search(comp_phase, arrayCurvature, arrayXcenter, arrayYcenter, Lambda, X, Y, dx, dy, sign, false);

            if (sum_max_True > sum_max_False)
                sign = true;
            else
                sign = false;
            end

            %sign = true;

            disp("Sign of spherical phase factor: " + string(sign));

            step_g = max(M,N)*0.1;
            step_h = step_g;
            step_cur = 0.6*cur;

            current_point = [cur, g, h];

            [phaseCompensate, phi_spherical, current_value] = funs.TSM(comp_phase, current_point(1), current_point(2), current_point(3), Lambda, X, Y, dx, dy, sign);

%             % Create a figure with three subplots
%             figure;
% 
%             % Plot the first image in the first subplot
%             subplot(1, 3, 1);
%             imagesc(angle(phi_spherical));
%             title('phi_spherical');axis square
% 
%             % Plot the second image in the second subplot
%             subplot(1, 3, 2);
%             imagesc(angle(comp_phase));
%             title('comp_phase');axis square
% 
%             % Plot the third image in the third subplot
%             subplot(1, 3, 3);
%             imagesc(phaseCompensate);
%             title('phaseCompensate');axis square
% 
%             % Show the figure
%             sgtitle('Subplots');

            disp('Starting the semiheuristic search of the accurate compensating parameters');

            while true
    
                neighbors = [];
%                 for dex = [-3 -2 -1 0 1 2 3]
%                     for dey = [-3 -2 -1 0 1 2 3]
%                         for dez = [-3 -2 -1 0 1 2 3]
                for dex = [-2 -1 0 1 2]
                    for dey = [-2 -1 0 1 2]
                        for dez = [-2 -1 0 1 2]
                            if dex == 0 && dey == 0 && dez == 0
                                continue
                            end
                            neighbor = [current_point(1) + dez*step_cur, current_point(2) + dex*step_g, current_point(3) + dey*step_h];
                            neighbors = [neighbors; neighbor];
                        end
                    end
                end

                optimal_neighbor = current_point;
                optimal_value = current_value;
                for neighbor = neighbors'
        
                    [phaseCompensate, phi_spherical, neighbor_value] = funs.TSM(comp_phase, neighbor(1), neighbor(2), neighbor(3), Lambda, X, Y, dx, dy, sign);
        
                    if neighbor_value > optimal_value
                        optimal_neighbor = neighbor;
                        optimal_value = neighbor_value;
                        disp("Best neighbor found! " + string(optimal_value));
            
%                         % Create a figure with three subplots
%                         figure
%                         subplot(1,3,1);
%                         imagesc(angle(phi_spherical));
%                         title('phi_spherical');axis square
%                         subplot(1,3,2);
%                         imagesc(angle(comp_phase));
%                         title('comp_phase');axis square
%                         subplot(1,3,3);
%                         imagesc(phaseCompensate);
%                         title('phaseCompensate');axis square
            
                    end
                end
    
                if optimal_value > current_value
                    current_point = optimal_neighbor;
                    current_value = optimal_value;
                else
                    break
                end
    
                step_g = step_g*0.5;
                step_h = step_g;
                step_cur = step_cur*0.5;
    
            end

            phi_spherical = ((X - current_point(2)).^2 .* dx^2 ./ current_point(1)) + ((Y - current_point(3)).^2 .* dy^2 ./ current_point(1));
            phi_spherical = pi * phi_spherical / Lambda;

            if (sign)
                phi_spherical = exp(1i * phi_spherical);
            else
                phi_spherical = exp(-1i * phi_spherical);
            end

            phaseCompensate = comp_phase .* phi_spherical;

            disp('Phase compensation finished.');

          end
    end
end
