%This function takes in a data matrix Xrun as well the mean vectors mu0, mu1 
%and the covariance matrices sigma0, sigma1 estimated from the training data
%and produces a column vector guesses, corresponding to the ML rule for Gaussian vectors
%with different means and the same covariance matrix, which is referred to as 
%Linear Discriminant Analysis (LDA) in machine learning.
function guesses = LDA(Xrun,mu0,mu1,sigmapooled)

guesses = []; % initializing it

for i = 1:size(Xrun)
    currentdata = Xrun(i,:);     
    guesses(i) = ((currentdata - mu0) * pinv(sigmapooled) * (currentdata - mu0)') >= ((currentdata - mu1) * pinv(sigmapooled) * (currentdata - mu1)');
end

guesses = guesses';

end