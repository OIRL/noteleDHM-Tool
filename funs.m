classdef funs
    methods(Static)

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

            figure; imagesc(BW);axis square; colormap gray;

          end
          
          function [plus_coor,m,n,p,q] = get_plus1(bw) 
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
          
          function holoCompensate = filter_center_plus1(FT_holo,plus_coor,m,n,Lambda,X,Y,dx,dy,k) 
            [M,N] = size(FT_holo);
            Filter = zeros(M,N);
            warning('off','all')
            Filter((plus_coor(2) - n) :(plus_coor(2)+ n) ,(plus_coor(1) - m) :(plus_coor(1)+ m))=1;
            FT_FilteringH = FT_holo.*Filter;

            figure,imagesc(log((abs(FT_FilteringH).^2))),colormap(gray),title('FT Hologram filter'),daspect([1 1 1]) 

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

          function [g,h] = get_g_and_h(bw) 

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
            figure, imshow(bw);
            axis on;
            hold on;
            plot(true_center(1), true_center(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
            plot(block_center(1), block_center(2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);
            line([true_center(1),block_center(1)],[true_center(2),block_center(2)],'Color','red','LineStyle','--')
            rectangle('Position', terms(best_term_idx).BoundingBox, 'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');

            % values for g and h
            g = g_and_h(1);
            h = g_and_h(2);
            disp("G: "+string(g)+" H: "+string(h));
          end

          function [J] = bin_CF_noTele_BAR_1d(fun,seed_cur,holoCompensate,M,N)
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
          
          function [J] = std_CF_noTele_BAR_1d(fun,Cx,holoCompensate,M,N)
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

    end
end