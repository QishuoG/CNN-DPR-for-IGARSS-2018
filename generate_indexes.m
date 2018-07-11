function [xlist,ylist,indexes]=generate_indexes(Itrain,groundtruth,patch,numImagesPerCategory,numClasses)


%% prepare training and validation data sets
 % context area m
channel=size(img,3);
selectedChannels = 1:channel; % Selected color channels
imageChannels = length(selectedChannels);


[xlist,ylist,indexes] = sampleImages(Itrain, patch, numImagesPerCategory, groundtruth, numClasses,1);
