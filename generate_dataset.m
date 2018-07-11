function [dataset]=generate_dataset(img,gt,patch,xlist,ylist)
I=img;
L=gt;
halfImg = floor(patch/2);
channel=size(img,3);
selectedChannels = 1:channel; % Selected color channels
imageChannels = length(selectedChannels);
train_images = zeros(patch, patch, imageChannels, numel(xlist),'single');
labels = zeros(1,numel(xlist), 'double');
for i=1:numel(xlist)
    x = xlist(i);
    y = ylist(i);
    im = 1;
    train_images(:,:,:,i) = I(x - halfImg: x + halfImg, y - halfImg: y + halfImg, :, im);
    labels(i) = L(x,y);
end

images = reshape(train_images,[],size(train_images,4));

% Split into training and validation sets
seed = 10;

[trainimages,valimages] = split(images,[0.9 0.1]);
[trainlabels,vallabels] = split(labels,[0.9 0.1]);% corresponding labels
% Reshape to 4D: dim x dim x channels x N

im_patches_4d=[trainimages valimages];
im_patches_4d=reshape(im_patches_4d, patch, patch, imageChannels, []);
% id of patches, id is the number of pixels in images
dataset.images.id=zeros(1,size(images,2));

dataset.images.id=1:size(labels,2);
% 4d image patches
dataset.images.data=im_patches_4d;

% which set: training-1,validation-2
dataset.images.set=zeros(1,size(images,2));
dataset.images.set(1:size(trainimages,2))=1;
dataset.images.set(size(trainimages,2)+1:size(trainimages,2)+size(valimages,2))=2;
dataset.images.label=[trainlabels vallabels];