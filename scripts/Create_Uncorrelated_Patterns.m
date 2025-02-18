% Parameters

num_bits = 72; % Number of bits per pattern
H = generate_hadamard(num_bits); %generates hadamard matrix of size [N,N]
valid = verify_hadamard(H); % verifies if patterns are orthogonal


%% --- Function Definitions ---

function [H, valid] = generate_hadamard(N)
    % GENERATE_HADAMARD Generates a Hadamard matrix of order N using Paley constructions
    %
    % Usage:
    %   [H, valid] = generate_hadamard(N)
    %
    % Inputs:
    %   N - The desired order of the Hadamard matrix
    %
    % Outputs:
    %   H    - The generated Hadamard matrix (or empty matrix if invalid N)
    %   valid - Boolean indicating if a valid Hadamard matrix was generated
    %
    % Limitations:
    %   1. Paley Type I Construction:
    %      - N-1 must be prime
    %      - N-1 must be congruent to 3 (mod 4)
    %      - Examples: N = 8, 12, 20, 28, 36, ...
    %
    %   2. Paley Type II Construction:
    %      - N/2 = p^m + 1 where p is an odd prime and m is positive integer
    %      - Examples: N = 12, 20, 28, 36, 44, ...
    %
    %   3. General limitations:
    %      - Not all Hadamard matrix orders can be constructed using Paley methods
    %      - For some valid N, the function might still fail due to computational limits
    
    % Input validation
    if ~isscalar(N) || ~isnumeric(N) || N < 1 || mod(N, 1) ~= 0
        error('Input N must be a positive integer.');
    end
    
    % Try Paley Type I first
    [H, valid] = try_paley_type1(N);
    
    % If Paley Type I fails, try Paley Type II
    if ~valid
        [best_k, best_p, best_m] = find_best_parameters(N);
        H = compute_Paley_type_2(best_k, best_p, best_m);
    end
end

function [best_k, best_p, best_m] = find_best_parameters(N)
    % Initialize the best parameters and the minimum error
    best_k = 0; best_p = 0; best_m = 0;
    min_error = inf; 
    
    % Upper bound for k is log2(N)
    max_k = floor(log2(N));
    
    % Iterate over possible k values (starting from 1)
    for k = 1:max_k
        % Calculate the reduced value N_reduced = N / 2^k
        N_reduced = N / (2^k);
        
        % Iterate over possible odd prime numbers for p (reasonable upper bound)
        primes_list = primes(floor(N_reduced - 1));
        primes_list = primes_list(primes_list > 2); % Only odd primes
        
        for p = primes_list
            % Solve p^m + 1 = N_reduced for m
            for m = 1:ceil(log(N_reduced) / log(p))
                value = p^m + 1;
                
                % Check if the value matches N_reduced
                if value == N_reduced
                    error_value = abs(N - 2^k * (p^m + 1));
                    if error_value < min_error
                        best_k = k;
                        best_p = p;
                        best_m = m;
                        min_error = error_value;
                    end
                end
            end
        end
    end
    
    % Display results
    if min_error < inf
        fprintf('Best parameters found: k = %d, p = %d, m = %d\n', best_k, best_p, best_m);
    else
        fprintf('No suitable parameters found for N = %d.\n', N);
    end
end

function  [Had,check] = compute_Paley_type_2(k,p,m)
    N_eff = 2^k * (p^m + 1);
    prim_poly = gfprimdf(m,p);
    [field2,expform] = gftuple((-1:p^m-2)',prim_poly,p);
    
    [~,~] = gftuple(prim_poly,m,p) ;
    field =field2;
    polyresidues = [];
    for i =1:p^m
        for j =1:p^m
            if (i~=j)
                sub_ij = gfsub(expform(i),expform(j), field);
                sqr_ij = gfmul(sub_ij,sub_ij, field);
                polyresidues = vertcat(polyresidues,sqr_ij);
    
            end 
    
        end 
    end 
    N=p^m+1;
    H = ones(N,N);
     
        for i=2:N
            H(i,i) =-1;
        end
        
        for i=2:N
            for j=2:N
                if ~ismember(gfsub(expform(j-1),expform(i-1), field),polyresidues)
                    H(i,j) = -1; 
                    
                end
    
            end
        end 
    q=p^m;

    if (mod(q,4) == 3)

    % Determine quadratic residues
        qres = mod((1:N_eff-2).^2, (N_eff-1));
    
    H = ones(N_eff,N_eff);
    
    for i=2:N_eff
        H(i,i) =-1;
    end
    
    for i=2:N_eff
        for j=2:N_eff
            if (j > i)
                if ~ismember(j-i,qres)
                    H(i,j) = -1; 
                else 
                    H(j,i) = -1;
                end
            end
        end
    end 

    Had = H;
    check = ((Had.' * Had)/(length(Had))) == eye(length(Had));
    if sum(sum(check))== length(Had)^2
        disp("patterns orthogonal")
    else
        disp("patterns not orthogonal")
    end   
    
    else
        Q = zeros(q,q);
        
        for i=1:q
            for j=1:q
                if (i ~= j)
                    if ismember(gfsub(expform(j),expform(i), field),polyresidues)
                        Q(i,j) = 1;
                    else
                        Q(i,j) = -1; 
                    end
                end 
            end
        end
    
        if k == 1
            B= ones(N,N);
            B(1,1) = 0 ; 
            B(2:N,2:N) = Q ; 
            Had = kron(B,hadamard(2)) + kron(eye(N),rot90(fliplr(-hadamard(2)),-1));
             
            check = ((Had.' * Had)/(length(Had))) == eye(length(Had));
            if sum(sum(check))==length(Had)^2
                disp("patterns orthogonal")
            else
                Had = kron(H,hadamard(2^(k-1)) ); 
                check = ((Had.' * Had)/(length(Had))) == eye(length(Had));
                if sum(sum(check))== length(Had)^2
                    disp("patterns orthogonal")
                else
                    disp("patterns not orthogonal")
                end        
            end 
            
        end
    end
end

function [H, valid] = try_paley_type1(N)
    % Implementation of Paley Type I construction
    valid = false;
    H = [];
    
    % Check if N-1 is prime and ≡ 3 (mod 4)
    if ~isprime(N-1) || mod(N-1, 4) ~= 3
        return;
    end
    
    % Calculate quadratic residues modulo N-1
    qres = mod((1:(N-2)).^2, N-1);
    qres = unique(qres);
    
    % Initialize Hadamard matrix
    H = ones(N);
    
    % Set diagonal elements to -1
    H(2:end, 2:end) = diag(-ones(1, N-1));
    
    % Fill the matrix using quadratic residues
    for i = 2:N
        for j = i+1:N
            diff = mod(j-i, N-1);
            if ~ismember(diff, qres)
                H(i,j) = -1;
                H(j,i) = 1;
            else
                H(i,j) = 1;
                H(j,i) = -1;
            end
        end
    end
    
    % Verify orthogonality
    valid = verify_hadamard(H);
end
function residues = calculate_field_residues(field, p, m)
    % Calculate quadratic residues in the finite field
    q = p^m;
    residues = zeros(1, q-1);
    idx = 1;
    
    for i = 1:q
        element = field(i,:);
        square = gfmul(element, element, field);
        residues(idx) = bi2de(square, 'left-msb');
        idx = idx + 1;
    end
    
    residues = unique(residues);
end

function core = construct_core_matrix(field, residues, size)
    % Construct the core matrix for Paley Type II
    core = ones(size);
    
    % Set diagonal to -1
    core(2:end, 2:end) = diag(-ones(1, size-1));
    
    % Fill the matrix using field residues
    for i = 2:size
        for j = 2:size
            if i ~= j
                diff = bi2de(gfsub(field(j-1,:), field(i-1,:), field), 'left-msb');
                if ~ismember(diff, residues)
                    core(i,j) = -1;
                end
            end
        end
    end
end

function valid = verify_hadamard(H)
    % Verify if matrix is a valid Hadamard matrix
    N = size(H, 1);
    tolerance = 1e-10;  % Numerical tolerance for floating point comparisons
    
    % Check orthogonality
    test = (H' * H) / N;
    valid = all(all(abs(test - eye(N)) < tolerance));
end