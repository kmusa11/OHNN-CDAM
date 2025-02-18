%% make the patterns
P= 20 ; %number of patterns
N=100; %number of pixels

corr_initial = 0.20;
bound = 0;
while bound < corr_initial*0.9999 || bound > corr_initial*1.0001
    m = zeros(1,P); % magnetization of each pattern
    d = eye(P) + corr_initial * (ones(P,P)- eye(P));
    g = horzcat(vertcat(d,m),[m';1]) ;
    h = sin(pi*g/2);
    mu =zeros(1,P+1);
    Sigma = h ;
    A = mvnrnd(mu,Sigma,N); % gaussian noise
    b = sign(A);
    c = zeros(N,P);
    for i =1:N
        c(i,:) = b(i,1:P)*b(i,P+1);
    end
    
    for z = 1:P
        m(z) = sum(c(:,z))/N;
    end
    
    % calculate the correlations
    corr = zeros(P,P);
    for k=1:P
        for l= 1:P
            corr(k,l) = (1/N*(c(:,k)' * c(:,l) ) - m(k)*m(l) )/sqrt((1 - m(k)^2)*(1 - m(l)^2));
        end
    end
    bound = sum(sum(corr-eye(P)))/(P^2-P) % insanity check !
end

%% Verify Average Correlation between Patterns

patterns = zeros(xeff*yeff,K);
for k = 1:K
    patterns(:,k) = 2*reshape(A_digit(:,yeff*(k-1)+1:yeff*k),[1,xeff*yeff])-1;
end

corr = calculate_correlation_matrix(patterns);
bound = sum(sum(corr-eye(K)))/(K^2-K) % insanity check !

%% Function Space %%

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