% D: d*n matrix of column vectors
% Normalize the columns to have L2 norm = 1
function D = l2norm(D)
dnorm = sqrt(sum(D.^2,1));
D = D./repmat(dnorm, size(D,1), 1);
end