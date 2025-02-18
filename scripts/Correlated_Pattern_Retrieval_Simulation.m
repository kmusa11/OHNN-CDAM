% Hopfield Network Analysis with Pattern Correlation
% This script analyzes pattern storage and retrieval in Hopfield networks
% with both 2-body and 4-body interactions

%% Parameters
N = 100;           % Number of neurons
xeff = 10;         % X dimension
yeff = 10;         % Y dimension
p_blurry = 0.40;   % Noise level for pattern corruption
max_patterns = 45; % Maximum number of patterns to test

% Create directories for saving results
if ~exist('results', 'dir')
    mkdir('results')
    mkdir('results/figures')
    mkdir('results/data')
end

%% Run Experiments

% Phase Transition Analysis
step_size = 0.05;
correlation_range = 0:step_size:0.0;
accuracies_2body = zeros(length(correlation_range), max_patterns);
accuracies_4body = zeros(length(correlation_range), max_patterns);

% Run experiments
for i = 1:length(correlation_range)
    desired_correlation = correlation_range(i);
    %[patterns, ~] = generate_patterns(max_patterns, N, desired_correlation); % Accurate correlation across patterns
    %[patterns, ~] = generate_correlated_patterns(N, max_patterns, desired_correlation); %Varying correlation across pattterns
    
    for num_patterns = 1:max_patterns
        % Test 2-body retrieval
        %accuracies_2body(i,num_patterns) = test_retrieval(patterns, num_patterns, ...
        %    N, xeff, yeff, p_blurry, false);
        
        % Test 4-body retrieval
        accuracies_4body(i,num_patterns) = test_retrieval(patterns', num_patterns, ...
            N, xeff, yeff, p_blurry, true);
    end
    
    fprintf('Completed correlation %.3f\n', desired_correlation);
end

%% Save Results
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% Save data
save(fullfile('results/data', sprintf('hopfield_results_%s.mat', timestamp)), ...
    'accuracies_2body', 'accuracies_4body', 'correlation_range', 'max_patterns', ...
    'N', 'p_blurry');

% % Plot and save 2-body results
% figure('Position', [100, 100, 800, 600]);
% imagesc(1:max_patterns, correlation_range, accuracies_2body);
% xlabel('Number of Patterns');
% ylabel('Correlation');
% title('2-Body Phase Transition');
% colorbar;
% savefig(fullfile('results/figures', sprintf('2body_phase_%s.fig', timestamp)));
% saveas(gcf, fullfile('results/figures', sprintf('2body_phase_%s.png', timestamp)));

% Plot and save 4-body results
figure;
imagesc(1:max_patterns, correlation_range, accuracies_4body);
xlabel('Number of Patterns');
ylabel('Correlation');
title('4-Body Phase Transition');
colorbar;
saveas(gcf, fullfile('results/figures', sprintf('4body_phase_%s.png', timestamp)));

%% Function Definitions
function accuracy = test_retrieval(patterns, num_patterns, N, xeff, yeff, p_blurry, is_four_body)
    num_runs = 50;
    success_count = 0;
    
    for run = 1:num_runs
        for p = 1:num_patterns
            % Add noise to pattern
            permvec_eff = randperm(xeff*yeff);
            s1_eff = floor(p_blurry * (xeff*yeff));
            s2_eff = xeff*yeff - s1_eff;
            svec_eff = [ones(1,s2_eff), zeros(1,s1_eff)];
            svec_eff = svec_eff(permvec_eff);
            corrupted_pattern = (2*svec_eff-1) .* patterns(p,:);
            
            % Retrieve pattern
            if is_four_body
                success = retrieve_four_body(corrupted_pattern, patterns, num_patterns, N);
            else
                success = retrieve_two_body(corrupted_pattern, patterns, num_patterns, N);
            end
            
            if success
                success_count = success_count + 1;
            end
        end
    end
    accuracy = success_count / (num_runs * num_patterns);
end

function success = retrieve_two_body(corrupted_pattern, patterns, num_patterns, N)
    % Calculate weight matrix
    W = patterns(1:num_patterns,:)' * patterns(1:num_patterns,:);
    W = W - diag(diag(W));  % Remove self-connections
    
    % Retrieval process
    retrieved_pattern = corrupted_pattern;
    max_iterations = 300;
    
    for iter = 1:max_iterations
        for i = 1:N
            net_input = W(i,:) * retrieved_pattern';
            retrieved_pattern(i) = sign(net_input);
            if retrieved_pattern(i) == 0
                retrieved_pattern(i) = 1;
            end
        end
        
        % Check convergence
        overlap = sum(retrieved_pattern == patterns(1:num_patterns,:), 2)/N;
        if any(overlap >= 0.80)
            success = true;
            return;
        end
    end
    success = false;
end

function success = retrieve_four_body(corrupted_pattern, patterns, num_patterns, N)
    retrieved_pattern = corrupted_pattern;
    max_iterations = 1;
    
    % Initial energy calculation
    net_input_initial = calculate_four_body_energy(retrieved_pattern, patterns, num_patterns);
    
    for iter = 1:max_iterations
        for i = 1:N
            % Flip bit and calculate energy difference
            retrieved_pattern(i) = -retrieved_pattern(i);
            net_input = calculate_four_body_energy(retrieved_pattern, patterns, num_patterns);
            
            if net_input <= net_input_initial
                retrieved_pattern(i) = -retrieved_pattern(i);  % Flip back
            else
                net_input_initial = net_input;
            end
        end
        
        % Check convergence
        overlap = sum(retrieved_pattern == patterns(1:num_patterns,:), 2)/N;
        if any(overlap >= 0.85)
            success = true;
            return;
        end
    end
    success = false;
end

function energy = calculate_four_body_energy(state, patterns, num_patterns)
    energy = 0;
    for p = 1:num_patterns
        energy = energy + (patterns(p,:) * state')^4;
    end
end

%% Generate Correlated Patterns
function [patterns, corr] = generate_correlated_patterns(N, P, desired_correlation)
    m = zeros(1, P);
    
    % Generate correlation matrix
    d = eye(P) + desired_correlation * (ones(P,P) - eye(P));
    g = [d, m'; m, 1];
    h = sin(pi*g/2);
    
    % Generate patterns using multivariate normal distribution
    A = mvnrnd(zeros(1,P+1), h, N);
    b = sign(A);
    
    % Extract patterns
    patterns = zeros(N, P);
    for i = 1:N
        patterns(i,:) = b(i,1:P) * b(i,P+1);
    end
    patterns = patterns';
    
    % Calculate actual correlations
    corr = calculate_correlations(patterns);
end

function corr = calculate_correlations(patterns)
    [P, N] = size(patterns);
    m = mean(patterns, 2);
    corr = zeros(P, P);
    
    for k = 1:P
        for l = 1:P
            corr(k,l) = (1/N*(patterns(k,:) * patterns(l,:)') - m(k)*m(l)) / ...
                        sqrt((1 - m(k)^2)*(1 - m(l)^2));
        end
    end
end

function [patterns, corr] = generate_patterns(num_patterns, num_bits, desired_correlation)
    patterns = zeros(num_bits, num_patterns);
    
    % Generate first pattern randomly
    patterns(:, 1) = 2 * (randi([0, 1], num_bits, 1)) - 1;
    
    % Generate subsequent patterns with correlation control
    for i = 2:num_patterns
        [patterns(:, i), ~] = create_pattern(desired_correlation, num_bits, patterns(:, 1:i-1));
    end
    
    % Calculate final correlation matrix
    corr = calculate_correlations(patterns);
end


function [new_pattern, best_corr] = create_pattern(target_correlation, num_bits, existing_patterns)
    max_attempts = 4000; % Increased max attempts
    best_pattern = [];
    best_error = inf;
    tolerance = 0.005;  % Tightened tolerance
    
    % Initialize with random pattern
    current_pattern = 2 * (randi([0, 1], num_bits, 1)) - 1;
    best_pattern = current_pattern;
    
    for attempt = 1:max_attempts
        % Calculate current correlation
        temp_patterns = [existing_patterns, current_pattern];
        corr_matrix = calculate_correlations(temp_patterns);
        
        % Extract correlations with existing patterns
        n_existing = size(existing_patterns, 2);
        current_corrs = corr_matrix(1:n_existing, end);
        
        % Calculate error as maximum deviation from target
        max_error = max(abs(current_corrs - target_correlation));
        mean_error = mean(abs(current_corrs - target_correlation));
        std_error = std(current_corrs);
        
        % Combined error metric
        error = max_error + mean_error + std_error;
        
        if error < best_error
            best_error = error;
            best_pattern = current_pattern;
            best_corr = current_corrs;
            
            if error < tolerance
                break;
            end
        end
        
        % Adaptive bit flipping strategy with temperature
        temperature = (max_attempts - attempt) / max_attempts;
        base_flip_rate = 0.1;  % Base rate of bits to flip
        num_bits_to_flip = max(1, round(num_bits * base_flip_rate * temperature));
        
        % Create new pattern by flipping selected bits
        flip_indices = randperm(num_bits, num_bits_to_flip);
        new_candidate = current_pattern;
        new_candidate(flip_indices) = -new_candidate(flip_indices);
        
        % Probabilistic acceptance of worse solutions (simulated annealing)
        if rand < temperature
            current_pattern = new_candidate;
        else
            % Local search: only accept if it improves
            temp_patterns = [existing_patterns, new_candidate];
            new_corr_matrix = calculate_correlations(temp_patterns);
            new_corrs = new_corr_matrix(1:n_existing, end);
            new_max_error = max(abs(new_corrs - target_correlation));
            new_mean_error = mean(abs(new_corrs - target_correlation));
            new_std_error = std(new_corrs);
            new_error = new_max_error + new_mean_error + new_std_error;
            
            if new_error < error
                current_pattern = new_candidate;
            end
        end
    end
    
    if best_error > tolerance
        warning('Could not achieve exact target correlation. Best error: %.4f', best_error);
    end
    
    new_pattern = best_pattern;
end