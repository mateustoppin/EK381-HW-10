%This function takes in a data matrix Xrun as well the mean vectors mu0, mu1 
%and the covariance matrices sigma0, sigma1 estimated from the training data
%and produces a column vector guesses, corresponding to the ML rule for Gaussian vectors
%with different means and different covariance matrices, which is referred to as 
%Quadratic Discriminant Analysis (QDA) in machine learning.
function guesses = QDA(Xrun,mu0,mu1,sigma0,sigma1)

guesses = [];
for i = 1:size(Xrun)
    currentdata = Xrun(i,:);
    var1 = sum(log(eig(sigma1))) + ((currentdata - mu1)) * pinv(sigma1) * (currentdata - mu1)';
    var2 = sum(log(eig(sigma0))) + ((currentdata - mu0)) * pinv(sigma0) * (currentdata - mu0)';
    guesses(i) = var1 <= var2;
end
guesses = guesses';
end