%This function takes in a data matrix Xrun, mean vector mu, 
%eigenvector matrix V, and eigenvalues D, and dimension k. 
%It selects the k eigenvectors corresponding to the k largest
%eigenvalues, centers the data by subtracting mu, and projects
%the centered data to k dimensions by multiplying by the matrix
%of k eigenvectors.
function Xrun_reduced = dimensionality_reduction(Xrun,mu,V,D,k)

[sD,sI] = sort(diag(D),'descend');
eigenk = [V(:,sI(1:k))];
Xrun_reduced = (Xrun - mu) * eigenk;

end