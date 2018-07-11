clc
clear
close all

rng('default')
setup()

load PU
patch=7;
img=pavia_corrected;
gt=groundtruth;
[m ,n, bands]=size(pavia_corrected);

img=mat2gray(img);
img=single(img);
% take average mean of each channel
mean_img=mean(mean(double(img),1),2);
for i=1:bands
    img(:,:,i)=img(:,:,i)-mean_img(1,1,i);
end

img_sub=[];

for i=1:bands
    
    img_sub(:,:,i)=imresize(img(:,:,i),0.25);
end

numClasses = 9;

numImagesPerCategory=zeros(1,9);
for i=1:9
    numImagesPerCategory(i)=min([length(find(gt==i)),200]);
end



channel=size(img,3);
selectedChannels = 1:channel; % Selected color channels
imageChannels = length(selectedChannels);


[xlist,ylist,indexes] = sampleImages(img, patch, numImagesPerCategory, gt, numClasses,1);

[dataset]=generate_dataset(img,gt,patch,xlist,ylist);

net=train_pu_patch(dataset);
s=[];
dim=floor(patch/2);

Itest=padarray(img,[dim,dim],'symmetric','both');

result_map=zeros(m,n);
score=zeros(m,n);

for i=1:m*n
    [X,Y]=ind2sub(size(gt),i);
    X_new = X+dim;
    Y_new = Y+dim;         
    X_range = X_new-dim : X_new+dim;
    Y_range = Y_new-dim : Y_new+dim;
    temp=Itest(X_range,Y_range,:);
    net.conserveMemory=0;
    net.eval({'input',temp});

%obtain the CNN output

    res=net.vars(net.getVarIndex('prediction')).value;
    scores=squeeze(gather(res));

%%%%%%show the classification results
    s=[s scores];
    [score(X,Y),result_map(X,Y)]=max(scores);
%       
   
end
prob=s;
prob=double(prob);
[acc] = ComputeClassificationAccuracy(result_map,gt)

%%%%%%%%%%%DPR%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_bands=103;
load Pulistall
list=PUlistall;
GIh = zeros(size(img));
GIv = zeros(size(img));
for i = 1:no_bands
    GIh(:,:,i) = (edge(img(:,:,i),'sobel','horizontal'));
    GIv(:,:,i) = (edge(img(:,:,i),'sobel','vertical'));
end
GradIm = 0.5*(exp(-sum(GIh,3))+exp(-sum(GIv,3)));
GradIm = GradIm/max(max(GradIm));
lambda = 0.85;
Kmax = 50;

[p_MLRpr,errmlr_MLR] = DPR(prob,GradIm,list,0.85,Kmax,0);
[maxMLRpr,MLRprmap] = max(p_MLRpr);
CNN_DPR_map=reshape(MLRprmap,610,340); % the result map for JSPR
[CNN_DPR_acc]=ComputeClassificationAccuracy(CNN_DPR_map,gt)