%% Simulating Dense Associative Memory in OHNN
%  Last Modified 01/12/2025
%  Khalid Musa


% initial parameters

N_spin=880; % Number of spins/neurons
N_macro=80; % Number of spins in a single Macro pixel 
macro = N_spin/ N_macro ; % Number of macro pixels
spin_scaling = 4; 
macro1  = macro*spin_scaling ;

s_size=get(0,'MonitorPositions');
screen_center_h = s_size(2,3)/2  ;
screen_center_v = s_size(2,4)/2;

% the number of spin is macro*macro = 11*11 = 121 
% each one is a macropixel j =1,...,macro*macro
% Fix a macropixel j , then this is compososed of N_macro * N_macro pixels 
% inside there are binary patterns 0,1 with a probability p_j of occuring
% imagine Monte Carlo darts falling on the macropixel target

%% Initialize DAQ

daqlist("ni");
dq = daq("ni");
dq.Rate = 8000;
addinput(dq, "Dev1", "ai0", "Voltage");
addinput(dq, "Dev1", "ai1", "Voltage");
addinput(dq, "Dev1", "ai2", "Voltage");

%% Remove DAQ

clear d
clear dq
clear ch
clear data

%% creating the effective region of beam 

x1 = 9;
x2 = 18;
y1 = 6;
y2 = 15;
xeff = x2 - x1 +1;
yeff = y2 - y1 +1;


%% creating the effective region of beam 

x1 = 10;
x2 = 18;
y1 = 7;
y2 = 14;
xeff = x2 - x1 +1;
yeff = y2 - y1 +1;

%% Create MNIST Digits

% Load data
load("mnist.mat")
load("mnist_padma.mat")

% Create digits structure using a loop
for i = 0:19
    baseIdx = mod(i, 10);
    eval(['digits.digit' num2str(i) ' = reshape(double(digit' num2str(baseIdx) '(:,:,' num2str( ceil((i+1)/10 )) ')), [28,28]);']);
end

% MNIST binary images rescaled
K = 15;

A = ones(macro,K*macro);
phase_amplitude = repelem(acos(2*A-1),spin_scaling,spin_scaling); 

A_digit = double(amp_mnist_eff(14, 14, digits, K) >= 0.4*max(max(A)));
phase_amplitude_digit = acos(2*A_digit-1);

for k=1:K
   phase_amplitude(x1:x2,(k-1)*macro1+y1:(k-1)*macro1+y2) = phase_amplitude_digit(:,(k-1)*yeff+1:k*yeff);
end

%% Convert variable correlated patterns to phase V1

K=1; % Number of Patterns
A = ones(macro,K*macro);
phase_amplitude = repelem(acos(2*A-1),spin_scaling,spin_scaling); 
A_digit = zeros(xeff,K*yeff);

for k = 1:K
    A_digit(:,(k-1)*yeff+1:k*yeff) = reshape(c(:,k),[xeff,yeff]);
end

phase_amplitude_digit = acos(A_digit);
A_digit = (A_digit+1)./2;

for k=1:K
    phase_amplitude(x1:x2,(k-1)*macro1+y1:(k-1)*macro1+y2) = phase_amplitude_digit(:,(k-1)*yeff+1:k*yeff);
end

%% Convert uncorrelated patterns to phase

K=length(H)-1; % Number of Patterns
A = ones(macro,(K+1)*macro);
phase_amplitude = repelem(acos(2*A-1),spin_scaling,spin_scaling);
A_digit = zeros(xeff,K*yeff);

for k = 2:K+1
    A_digit(:,(k-2)*yeff+1:(k-1)*yeff) = reshape(H(:,k),[xeff,yeff]);
end

phase_amplitude_digit = acos(A_digit);
A_digit = (A_digit+1)./2;

for k=1:K
    phase_amplitude(x1:x2,(k-1)*macro1+y1:(k-1)*macro1+y2) = phase_amplitude_digit(:,(k-1)*yeff+1:k*yeff);
end

%% Verify average correlation of any set of patterns

patterns = zeros(xeff*yeff,K);
for k = 1:K
    patterns(:,k) = 2.*reshape(A_digit(:,yeff*(k-1)+1:yeff*k),[1,xeff*yeff])-1;
end

corr = calculate_correlation_matrix(patterns);
bound = sum(sum(corr - eye(K)))/(K^2 - K);
disp(strcat("Average Correlation = ",num2str(bound)))

%% run it to create one spin configuration [macro1*macro1]

spin_scaling = 4; 
macro1 = macro*spin_scaling ;

p_spin = 1/2; % Probability of spins up
s1 = floor(p_spin * (macro1*macro1));
s2 = macro1*macro1 - s1; 

%spin vector
svec = [ones(1,s1),zeros(1,s2)]; 
svec = svec(randperm(macro1*macro1)); 

%convert from {0,1} to {-pi/2,+pi/2}
phase_spin = phase_spin_conversion(svec);

%% Making the besselJ function

delete(findall(0,'type','figure'));

x = linspace(0,1,N_spin);
fx = besselj(1,x); 

[max_fx,max_i] = max(fx);
max_x = x(max_i);
[min_fx,min_i] = min(fx);
min_x = x(min_i);

x = linspace(x(min_i),x(max_i),length(x));
fx = besselj(1,x);

%% Create the checkerboard [macro1,macro1]

spin_scaling =4; 
checkersfull = ones(1,macro1*macro1) ;
    for j=1:1:macro1*macro1 
        if mod(ceil(j/macro1),2) == 1
            checkersfull(j) = (-1)^j;
        else
            checkersfull(j) = (-1)^(j+1);
        end
    end    
checkers = reshape(checkersfull,[macro1,macro1]);

%% creates the grating and F = f(A) inverse bessel used for Complex amplitude modulation
nx = 150;
ny = 0;
Amp = repelem(A(:,1:macro),N_macro,N_macro); 
[V,H] = size(Amp) ;

vA=abs(Amp)/max(abs(Amp),[],'all'); % normalized to unity 
aux = round(vA *N_spin);
y = linspace(-(V/2),(V/2)-1,V);
x = linspace(-(H/2),(H/2)-1,H);
x = x*10.4e-6; % Scales the hologram in the V direction
y = y*10.4e-6; % Scales the hologran in the H direction
[X,Y] = meshgrid(x,y); 
F = zeros(V,H);
      for mh = 1:V
          for nh = 1:H
              temp = aux(mh,nh);
              F(mh,nh) = fx(temp);                                
          end                                                                    
      end 
% Blaze grating periods/frequencies to shift the modulated beam to higher orders  
gy = ny/(V*10.4e-6);
gx = nx/(H*10.4e-6);
grating = 2*pi*(X*gx+Y*gy);

%% Energy landscape sweeping spin probability
permus = 1;
[Powersweep_all, t2sweep_all] = deal([]);

% Calculate effective parameters
[x1_eff, x2_eff, y1_eff, y2_eff] = deal(((x1-1)*N_macro)/spin_scaling, ...
    ((x2)*N_macro)/spin_scaling, ((y1-1)*N_macro)/spin_scaling, ((y2)*N_macro)/spin_scaling);
params = [x1_eff, x2_eff, y1_eff, y2_eff];
[xeff, yeff] = deal(x2-x1+1, y2-y1+1);

% Initialize spin parameters
spin_scaling = 4;
macro1 = macro*spin_scaling;
N_macro1 = N_spin/macro1;
p_spin = 1/2;
[s1, s2] = deal(floor(p_spin*(macro1^2)), macro1^2-floor(p_spin*(macro1^2)));
svec = [ones(1,s1), zeros(1,s2)];
svec = svec(randperm(macro1^2));

% Main permutation loop
for permutation = 1:permus
    permvec_eff = randperm(xeff*yeff);
    [Powersweep, t2sweep, blackflag] = deal([], [], 0);
    
    % Probability sweep
    for proba = 0:0.01:1
        [t1, t2] = deal(1, 100); %2-body and 4-body coefficients
        blackflag = blackflag + 1;
        
        % Calculate spin vectors
        s1_eff = floor(proba*(xeff*yeff));
        svec_eff = [ones(1,s1_eff), zeros(1,xeff*yeff-s1_eff)];
        svec_eff = svec_eff(permvec_eff);
        
        % Reshape and update svec
        svec_2d = reshape(svec, [macro1,macro1]);
        svec_2d(x1:x2,y1:y2) = reshape(svec_eff,[xeff,yeff]);
        svec = svec_2d(:)';
        
        % Phase calculations
        phase_slm = repelem(phase_screen(N_spin,spin_scaling,svec,checkers,A),N_macro1,N_macro1);
        grating_eff = zeros(size(grating));
        grating_eff(x1_eff:x2_eff,y1_eff:y2_eff) = grating(x1_eff:x2_eff,y1_eff:y2_eff);
        
        % Hologram processing
        Hol = F.*mod(phase_slm+grating_eff,2*pi);
        mask = Hol/max(Hol(:))*1023;
        putmiddle_phase_slm = put_middle(mask,s_size,screen_center_h,screen_center_v);
        
        % Data acquisition and processing
        Send_SLM_pump(SLM_colormap,putmiddle_phase_slm,s_size);
        pause(0.75);
        [Signal, SHG, ~] = read_oscilloscope(dq);
        mean_data = t1 * Signal + t2 * SHG;
        [power, t2] = deal(mean_data, Signal/SHG);
        [t2sweep, Powersweep] = deal([t2sweep, t2], [Powersweep, power]);
        close;
    end
    [Powersweep_all, t2sweep_all] = deal([Powersweep_all; Powersweep], [t2sweep_all; t2sweep]);
end

% Plot results
figure; hold on;
arrayfun(@(j) plot(Powersweep_all(j,:)), 1:permus);
hold off;

figure; hold on;
arrayfun(@(j) plot(t2sweep_all(j,:)), 1:permus);
hold off;

%% Find effective region of laser on SLM of size -> [xeff,yeff] = [10x10]

[xeff, yeff, x2, y2] = deal(9, 8, 11, 10);
[Previous_Power, final_params] = deal(0, zeros(1,4));

% Initialize spin vectors
[s1, s2] = deal(floor(p_spin*macro1^2), macro1^2-floor(p_spin*macro1^2));
svec = [ones(1,s1), zeros(1,s2)];
svec = svec(randperm(macro1^2));
permvec_eff = randperm(xeff*yeff);
N_macro1 = N_spin/macro1;

% Initialize progress bar
wb = waitbar(0, '');

% Main scanning loop
for i = y2-1:macro1
    for j = x2-1:macro1
        % Update progress bar
        p = ((i-1)*macro1 + j)/(macro1^2);
        waitbar(p, wb, sprintf('Ising Experiment: %.1f%%', p*100));
        
        % Calculate effective coordinates
        [x1, y1] = deal(j-xeff+1, i-yeff+1);
        [x1_eff, x2_eff, y1_eff, y2_eff] = deal(...
            ((x1-1)*N_macro)/spin_scaling, ...
            ((j)*N_macro)/spin_scaling, ...
            ((y1-1)*N_macro)/spin_scaling, ...
            ((i)*N_macro)/spin_scaling);
        
        % Generate spin vectors
        s1_eff = floor(1*xeff*yeff);
        svec_eff = [ones(1,s1_eff), zeros(1,xeff*yeff-s1_eff)];
        svec_eff = svec_eff(permvec_eff);
        
        % Update and reshape svec
        svec_2d = reshape(svec, [macro1,macro1]);
        svec_2d(x1:j,y1:i) = reshape(svec_eff,[xeff,yeff]);
        svec = svec_2d(:)';
        
        % Phase calculations and mask generation
        phase_spin = phase_spin_conversion(svec);
        phase_slm = repelem(phase_screen(N_spin,spin_scaling,svec,checkers,A),N_macro1,N_macro1);
        mask = GenerateMask(phase_slm,grating,F,[x1_eff,x2_eff,y1_eff,y2_eff]);
        putmiddle_phase_slm = put_middle(mask,s_size,screen_center_h,screen_center_v);
        
        % Send to SLM and measure power
        Send_SLM_pump(SLM_colormap, putmiddle_phase_slm, s_size);
        pause(0.75);
        [~, Power_All, ~] = read_oscilloscope(dq);
        
        % Update if new maximum found
        if Power_All > Previous_Power
            disp(Power_All)
            [Previous_Power, final_params] = deal(Power_All, [x1,j,y1,i]);
        end
        close
    end    
end
[x1,x2,y1,y2] = deal(final_params(1),final_params(2),final_params(3),final_params(4));
close(wb)

%% Monte Carlo Initial Parameters   

iterations = 72; % number of iterations
K=25; % number of stored patterns
Ifirst = 0;
t2 = 200; % scaling coefficient of the SHG power
n_rep = 25; % number of replicas
L = 2; % number of trials
Bluriness = 0.8; % Amount of masking
blur_vec = Bluriness * ones(1,L); % Initial bluriness = \delta


I_rep = zeros(n_rep*L,iterations+1) ;
Iall_rep = zeros(n_rep*L,(iterations+1) * K) ; 
Allspins_tem = zeros(n_rep*L*macro1^2, iterations+1);
P_unmod_rep = zeros(1, iterations*n_rep*L);
hamming_dist = zeros(n_rep,K);

estimated_run_time = iterations * K * n_rep * L * 0.7857 / 3600; % 785.7 ms/iteration          
disp(strcat("run time = ", num2str(estimated_run_time)," hours"))
%% Monte Carlo Run

tic
wb = waitbar(0,'');
[~, ~, Ref_I] = read_oscilloscope(dq); % reference intensity 

for i = 1: n_rep  
   
    p_spin=0.5;
    s1 = floor(p_spin * (macro1*macro1));
    s2 = macro1*macro1 -s1; 

    for l =1:L
        %effective spin vector
        p_blurry = blur_vec(l);
        s1_eff = floor(p_blurry* (xeff*yeff) ) ; 
        svec_eff = [ones(1,s1_eff),zeros(1,xeff*yeff -s1_eff)];
        svec_eff = svec_eff(randperm(xeff*yeff)) ;
        svec_eff_mat0 = reshape(svec_eff,[xeff,yeff]);

        %spin vector
        svec= [ones(1,s1),zeros(1,s2)]; 
        svec= svec(randperm(macro1^2));
        svec_mat = reshape(svec,[macro1,macro1]);

        %blur vector
        digit_section = A_digit(:,(i-1)*yeff+1:i*yeff);
        svec_eff_mat = svec_eff_mat0.*digit_section + (svec_eff_mat0<1/2).*(~digit_section);
        svec_mat(x1:x2,y1:y2) = svec_eff_mat ;
        svec = reshape(svec_mat,[1,macro1*macro1]);

        for a = 1:K
            hamming_dist(i,a) = sum(sum(svec_eff_mat~=A_digit(:,(a-1)*yeff +1 :a*yeff)));
        end

        Temp =  0;
        msg = sprintf('Ising Experiment in Progress: %.2f%% \n replica num = %.0f \n Randomness = %.2f %%',0,i,(1-blur_vec(l))*100);
        [I,I_all,Allspins,P_unmod] = montecarlo_oscillo(Ref_I,wb,i,blur_vec(l),l,dq,t2,s_size,Ifirst,svec,Temp,iterations,phase_amplitude,checkers,N_spin,K,grating,F,x1,x2,y1,y2);
        I_rep((i-1)*L+l,:) = I;
        Iall_rep(2*((i-1)*L+l) - 1: 2*((i-1)*L+l), :) = I_all; 
        Allspins_tem(((i-1)*L+l-1)*macro1^2 + 1 :((i-1)*L+l)*macro1^2, :) = Allspins;
        P_unmod_rep(1, ((i-1)*L+(l-1))*iterations + 1: ((i-1)*L+(l-1)+1)*iterations) = P_unmod;
    end     
 end

close(wb)
toc
%% Save Results
date_str = datestr(now, 'mm-dd-yy');
num_bodies = 4;
folder_name = sprintf('C:/Users/SLM2022/Desktop/Khalid/Dense Associative Memory/hybrid data/DAM Random Correlated Patterns/%d%% Correlation', round(bound*100));

filename = sprintf('%s-%dbod-%dP-%dR-%dB-%.1fRand-%.2fCorr-1.0RB-CORR.mat', ...
    date_str, num_bodies, K, n_rep, L, 1-Bluriness, bound);

% Save data
save(fullfile(folder_name, filename));

%% Intensity Plot
figure;
hold on;
colors = lines(n_rep * L); % Generate distinct colors for each curve
str = cell(1, min(n_rep * L, 20)); % Update legend to show up to 20 entries

for j = 1:n_rep * L
    % Plot each line with a distinct color and marker
    plot(I_rep(j, 2: end ), 'Color', colors(mod(j-1, size(colors, 1)) + 1, :), 'DisplayName', sprintf('Curve %d', j));
    
    % Update legend entries only for the first 20 curves to avoid clutter
    if j <= 20
        str{j} = sprintf('Curve %d', j);
    end
end

% Add legend, xlabel, ylabel, and title
legend(str, 'Location', 'southwest');
hold off;
xlabel('Monte Carlo iterations');
ylabel('Intensity of 2-body interaction');
title(sprintf('Intensity for %d replicas', n_rep));
grid on;

%% Intensity for Each Pattern
figure;
hold on;
colors = lines(K); % Generate distinct colors for each pattern
str = cell(1, K); % Initialize legend entries

% Plot each pattern with distinct colors and markers
for k = 1:K
    plot(Iall_rep(2, k:K:end), 'Color', colors(k, :), 'DisplayName', sprintf('Pattern %d', k));
    str{k} = sprintf('Pattern %d', k); % Update legend entries
end

% Add legend, xlabel, ylabel, and title
legend(str, 'Location', 'southwest');
hold off;
xlabel('Monte Carlo iterations');
ylabel('Intensity of Each Pattern');
title('Intensity for Each Pattern');
grid on;

%% Reference intensity for Laser fluctuations

figure;plot(P_unmod_rep)
legend({num2str(j)},'Location','southwest')
hold off
xlabel('Monte Carlo iterations') 
ylabel('Reference intensity for Laser fluctuations') 
title(sprintf('Reference intensity for %d replicas',n_rep))
fluctuation = (max(P_unmod_rep) - min(P_unmod_rep)) / mean(P_unmod_rep);
disp(strcat("average fluctuation of power = ", num2str(fluctuation*100),"%"))

hold off

%%

images = zeros(n_rep,N_spin,N_spin );
for j=1:n_rep
    replica = phase_spin_conversion(Allspins_tem((j-1)*121 +1 :j*121,end)) ; 
    images(j,:,:) = replica ; 

end 
ImageFolder='C:\Users\ising\Downloads\Ising_PhaseTransition-2024\Ising_PhaseTransition-2023\DenseIsing';
filename = "testAnimated.gif"; % Specify the output file name
gifName = fullfile(ImageFolder,filename) ;
s=sliceViewer(permute(images, [2, 3, 1]), 'ScaleFactors', [10, 10, 1]);

%Get the handle of the axes containing the displayed slice.

hAx = getAxesHandle(s);
%Specify the name of the GIF file you want to create.


%Create an array of slice numbers.

sliceNums = 1:size(images,1);

for idx = sliceNums
    % Update slice number
    s.SliceNumber = idx;
    % Use getframe to capture image
    I = getframe(hAx);
    [indI,cm] = rgb2ind(I.cdata,256);
    % Write frame to the GIF file
    if idx == 1
        imwrite(indI,cm,gifName,'gif','Loopcount',inf,'DelayTime', 0.2);
    else
        imwrite(indI,cm,gifName,'gif','WriteMode','append','DelayTime', 0.2);
    end
end

%% Post Processing Data

%% Create and filter Replicas (Final spin state)

replicas = zeros(n_rep*L,macro1*macro1);

for j=1:n_rep*L
    replicas(j,:) = Allspins_tem((j-1)*macro1*macro1 +1 :j*macro1*macro1,end); 
end 

replicas_mat_eff = zeros(n_rep*L,xeff,yeff);
replicas_eff = zeros(n_rep*L,xeff*yeff);
for j=1:n_rep*L
    rep_mat_j = reshape( replicas(j,:),[macro1,macro1]) ;  
    replicas_mat_eff(j,1:end,1:end) = rep_mat_j(x1:x2,y1:y2); 
    replicas_eff(j,:) = reshape(replicas_mat_eff(j,:,:),[1,xeff*yeff] );
end 

%% Visually compare initial/final spin state with ground truth

trial_num = 1;
for j = trial_num:L:n_rep*L
    figure;image(reshape(replicas_eff(j,:,:),[xeff,yeff] ),'CDataMapping','scaled')
    u = ceil(j/L);
    title(sprintf('Final state image K=%1.f',u))
    figure;imagesc(A_digit(:,yeff*(u-1) + 1: yeff*u));
    title(sprintf('Ground Truth image K=%1.f',u))
end
colorbar
%%
% Define the directory to save the files
saveDir = "C:\Users\SLM2022\Desktop\digits 12-15";

% Ensure the directory exists (create it if it doesn't)
if ~isfolder(saveDir)
    mkdir(saveDir);
end
count = 0;

% Loop through indices with step size L
for j = 1:L:n_rep*L
    % Plot the reshaped image from replicas_eff
    figure;
    image(reshape(replicas_eff(j, :, :), [xeff, yeff]), 'CDataMapping', 'scaled');
    
    % Remove axis ticks and labels for cleaner visualization
    axis off;

    % Optional: Add a colorbar
    % colorbar;

    % Generate a dynamic filename based on the iteration number
    filename = sprintf('Ground Truth Digit %d.png', count);
    count = count + 1;
    % Full path to save the file
    fullPath = fullfile(saveDir, filename);

    % Save the figure
    saveas(gcf, fullPath);

    % Optional: Close the figure after saving
    close;
end

%% m-parameter (overlap matrix)
Retrieve = zeros(1,L);
correlation = zeros(1,L);
temp1 = zeros(K,xeff*yeff);
threshold =  0.6; % use if patterns are orthogonal/hadamard
%threshold =  0.25; % use if patterns are correlated

% Reshape patterns
for k = 1:K
    temp1(k,:) = reshape(A_digit(:,yeff*(k-1)+1:yeff*k),[1,xeff*yeff]);
end

for i = 1:L
    breaking = 0;
    m = pattern_replica_overlap(temp1(1:end,:),replicas_eff(i:L:end,:));
    
    % Visualize overlap matrix
    figure;
    image(m,'CDataMapping','scaled')
    colorbar
    title(strcat('Confusion Matrix. Blurry = ',num2str(1-blur_vec(i))));
    
    % Analysis for each column
    for j = 1:width(m)
        % Store diagonal element for correlation
        correlation(1,i) = correlation(1,i) + m(j,j);
        
        % Get diagonal element for current column
        diag_val = m(j,j);
        
        % Get off-diagonal elements for current column
        off_diag = m(:,j);
        off_diag(j) = []; % Remove diagonal element
        
        % Check if diagonal is the maximum absolute value
        if abs(diag_val) == max(abs(m(:,j)))
            % Count how many off-diagonal elements are too close to diagonal
            too_close = sum(abs(off_diag) >= (abs(diag_val) * (1 - threshold)));
            
            % If no off-diagonal elements are too close, increment breaking
            if too_close == 0
                breaking = breaking + 1;
            end
        end
    end
    
    % Calculate retrieval accuracy for this pattern
    b = width(m);
    Retrieve(1,i) = breaking/b;
    correlation(1,i) = correlation(1,i)/b;
end

% Display results
disp(strcat("The average retrieval accuracy = ", num2str(mean(Retrieve))))

%% FUNCTIONs SPACE

%% time division multiplexing 

function [I_new,I_all,PM] = tdm_oscillo(params,Ref_I,dq,t2,s_size,N_macro1,macro1,phase_spin,phase_amplitude,checkers,K,grating,F)

% Preallocate variables
I_new = 0;
I_all = zeros(2,K) ; 
screen_center_h = s_size(2,3)/2;
screen_center_v = s_size(2,4)/2;

    % Main loop
    for k = 1:K
        % Calculate phase and apply checkerboard modulation
        phase = mod(phase_spin + checkers .* phase_amplitude(1:macro1, (k-1)*macro1 + 1 : k*macro1), 2*pi);
        phase_slm = repelem(phase, N_macro1, N_macro1); 
        
        % Generate SLM pattern with grating
        grate_phase_slm = GenerateMask(phase_slm, grating, F, params);
        putmiddle_phase_slm = put_middle(grate_phase_slm, s_size, screen_center_h, screen_center_v);

        % Send SLM pattern
        Send_SLM_pump(SLM_colormap, putmiddle_phase_slm, s_size);
        pause(0.6); % Synchronize SLM and PM

        % Read oscilloscope data
        [FM,SHG,PM] = read_oscilloscope(dq);

        % Close SLM Pattern
        close

        % Normalize and calculate intensities
        Norm = 1 + (PM - Ref_I) / Ref_I;
        PP1 = FM * Norm;
        SHP1 = SHG * (Norm^2);

        % Store results
        I_all(:, k) = [PP1; SHP1]; 
        I_new = I_new + PP1 + t2*SHP1;
    end
    disp(I_all)
end

%% Helper function to read oscilloscope data
function [data1,data2,data3] = read_oscilloscope(dq)
    data = read(dq, seconds(0.0475));
    data1 = mean(data.Dev1_ai0);
    data2 = mean(data.Dev1_ai1);
    data3 = mean(data.Dev1_ai2);    
end


%% Monte Carlo Spin Change Asynchronous 
function [I, I_all, Allspins, P_unmod] = montecarlo_oscillo(Ref_I,wb,i,blur_vec,l,dq,t2,s_size,Icurrent,svec,T,iterations,phase_amplitude,checkers,N_spin,K,grating,F,x1,x2,y1,y2)
   
    % Initialization and preallocation
    I = zeros(1, iterations + 1);
    I_all = zeros(2, K*iterations) ;
    P_unmod = zeros(1, iterations);
    Allspins = zeros(length(svec), iterations + 1); 
    Allspins(:, 1) = svec;
    I(1, 1) = Icurrent;

    macro1 = sqrt(length(svec));
    N_macro1 = N_spin/macro1; 
    svec_current = svec;
    svec_mat = reshape(svec, [macro1,macro1]); 

    xeff = x2 - x1 + 1;
    yeff = y2 - y1 + 1;
    svec_mat_eff = svec_mat(x1:x2, y1:y2) ; 
    x1_eff = (x1 - 1) * N_macro1;
    x2_eff = x2 * N_macro1;
    y1_eff = (y1 - 1) * N_macro1;
    y2_eff = y2 * N_macro1;
   
    svec_current_eff  = reshape(svec_mat_eff,[1,xeff*yeff] );
    
    for it = 2:iterations+1
        p = (it-1)/iterations;
        svec_new_eff = spinupdate2(svec_current_eff,mod(it-2,xeff*yeff)+1);  
        params = [x1_eff,x2_eff,y1_eff,y2_eff];

        svec_mat_new = reshape(svec_current,[macro1,macro1]); 
        svec_mat_new(x1:x2,y1:y2) = reshape(svec_new_eff,[xeff,yeff]);
        svec_new = reshape(svec_mat_new,[1,macro1*macro1]);

        spin_matrix = reshape (svec_new,[macro1,macro1]);
        phase_spin = (2*spin_matrix -1 )* pi/2 ;
    
        [Inew,I_all(:,(it-1)*K +1 : it*K ), P] = tdm_oscillo(params,Ref_I,dq,t2,s_size,N_macro1,macro1,phase_spin,phase_amplitude,checkers,K,grating,F);
        
        P_unmod(1, it - 1) = P; 
        msg = sprintf('Ising Experiment: %.1f%% \n (Replica,Randomness,#) = (%.0f, %.0f%%,%.f)',p*100,i,((1-blur_vec)*100),l);
        waitbar(p,wb,msg);

        deltaI = Inew - Icurrent; 
        sample_r = rand(1,1);
        if deltaI >= 0  || sample_r < exp(deltaI/T)
            disp('Accepted spin flip');
            Icurrent = Inew;
            svec_current_eff = svec_new_eff;
            svec_current = svec_new;
            I(1,it) = Inew ;
            Allspins(:,it) = svec_new;   
        else 
            I(1,it) = Icurrent;
            Allspins(:,it) = svec_current;
        end
    end
end


function Send_SLM_pump_New(z)
    %show_slm_preview(1.0);
    z = rot90(z,2);
    heds_show_data(double(z/255));
end

function Send_SLM_pump(color_map,z,s_size)
% s_size=get(0,'MonitorPositions'); %Get the screensize
% 
% % Defining figure position to be sent to SLM 
figure('Position',s_size(2,:),'MenuBar','none','ToolBar','none','resize','off') %fullscreen SLM

% %set_figure_toscreen(2)
image(z)
colormap(color_map)
%colormap(gray(1023));

%eleminating superflous feature of the image
axis off 
set(gca,'position',[0 0 1 1],'Visible','off')

%set(gcf,'position',[screen_center_h screen_center_v h v]) 
end

%% colormap for Santec SLM
function color_map=SLM_colormap
GrayScale=1024;
color_map=zeros(GrayScale, 3);
red=0;
blue=0;
green=0;
count=1;
for r=1:8
    for g=1:8
        for b=1:16
         color_map(count,1)=red;
         color_map(count,2)=green;
         color_map(count,3)=blue;
         blue=blue+16;
         count=count+1;
        end
        blue=0;
        green=green+32;
    end
    green=0;
    blue=0;
    red=red+32;
end
color_map=color_map/max(max(color_map));
end


%% takes matrix A and puts it in the middle of SLM screen, making background black 

function ret = put_middle(A,s_size,p_h,p_v)
    pxl_rows= s_size(2, 4); %2 is SLM , 1 is computer screen 
    pxl_cols = s_size(2, 3);
    x= zeros(pxl_rows,pxl_cols);
    v= size(A,1);
    h= size(A,2);
    x( (p_v - floor(v/2)): (p_v + floor(v/2) -1) , (p_h - floor(h/2)): (p_h + floor(h/2) -1)) = A; 
    ret = x ;
end 

%%
function A = amp_mnist_eff(xeff,yeff,digits,K)
   b = 10; c = 10;
   b1 = (xeff-b)./2;
   c1 = (yeff-c)./2;
   A = zeros(b,K*c);
   for k = 0:K-1 
       dname = sprintf('digit%u',k) ; 
       digit = digits.(dname) ;
       temp_A = imresize(digit, [xeff, yeff],'bicubic');
       A(:,k*c+1:(k+1)*c) = temp_A(1+round(b1):end-round(b1),1+round(c1):end-round(c1));
   end 
end 

%% spin matrix update

function newsvec = spinupdate(svec)
    newsvec = svec; 
    i = randi(length(svec)); %randomly picks spin site
    if (svec(i)==0)
        newsvec(i)=1;
    else 
        newsvec(i) = 0 ;
    end 
end 

function newsvec = spinupdate2(svec,i)
    newsvec = svec; %sequentially picks spin site 
    if (svec(i)==0)
        newsvec(i)=1;
    else 
        newsvec(i) = 0 ;
    end 
end 

function phase_spin = phase_spin_conversion(svec)
    macro1 = sqrt(length(svec));
    spins = reshape (svec,[macro1,macro1]);
    phase_spin = (2*spins-1) * pi/2;
end


function phase_slm = phase_screen(~,spin_scaling,svec,checkers,A)
    k=1;
    macro1 = sqrt(length(svec));
    spins = reshape (svec,[macro1,macro1]);
    
    % phase matrix
    phase_spin = (2*spins -1 )* pi/2 ;
    phase_amplitude = acos(2*A-1);
    phase_amplitude = repelem(phase_amplitude,spin_scaling,spin_scaling);
    
    phase = phase_spin + checkers .* phase_amplitude(1:macro1, (k-1)*macro1 +1 : k*macro1 );
    phase_slm =  mod(phase,2*pi);
    
end
 
function mask = GenerateMask(phase_slm,grating,F, params)
    x1_eff = params(1);
    x2_eff = params(2);
    y1_eff = params(3);
    y2_eff = params(4);
    phase_slm = mod(phase_slm,2*pi);
    grating_eff = zeros(size(grating));
    grating_eff(x1_eff:x2_eff,y1_eff:y2_eff) = grating(x1_eff:x2_eff,y1_eff:y2_eff);
    Hol = F.*mod(phase_slm+grating_eff,2*pi);
    mask = Hol/max(max(Hol))*1023;
end

%%
function m = pattern_replica_overlap(patterns,replicas)
    replicas = 2*replicas -1 ;
    patterns = 2*patterns -1 ; 
    N = size(patterns,2); 
    n_rep = size(replicas, 1 ) ; 
    n_patt = size(patterns, 1) ; 
    m = zeros(n_patt,n_rep);
    for i = 1:n_patt
        for j = 1:n_rep
        m(i,j) = 1/N * sum(patterns(i,:) .* replicas(j,:));
        end 
    end 
end 

function corr = calculate_correlation_matrix(patterns)
    N = size(patterns, 1);
    P = size(patterns, 2);
    
    % Calculate means for each pattern
    m = mean(patterns);
    
    % Initialize correlation matrix
    corr = zeros(P, P);
    
    % Calculate correlation using the original definition
    for k = 1:P
        for l = 1:P
            corr(k, l) = (1/N * (patterns(:, k)' * patterns(:, l)) - m(k) * m(l)) / ...
                         sqrt((1 - m(k)^2) * (1 - m(l)^2));
        end
    end
end

