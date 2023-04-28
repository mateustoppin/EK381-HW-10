%This function takes in a data matrix Xrun as well a training data matrix 
%Xtrain and a labeled output Ytrain and produces a column vector guesses,
%correspoding to the nearest neighbor classifier.  

function guesses = nearest_neighbor(Xrun,Xtrain,Ytrain)

dist = [];
guesses = [];

for i = 1:size(Xrun, 1)
    for k = 1:size(Xtrain, 1)
        dist(k)  = sum((Xrun(i,:) - Xtrain(k,:)).^2);
    end

    minimum_dist = min(dist);
    small_index = find(dist == minimum_dist);
    guesses(i) = [Ytrain(small_index)];
end

guesses = guesses';

end

