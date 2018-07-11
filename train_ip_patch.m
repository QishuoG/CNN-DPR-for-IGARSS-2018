function [net]=train_ip_patch(imdb)

%
run ('vl_setupnn.m');

%%%%%common options%%%%%%
trainOpts.batchSize=10;
trainOpts.numEpochs=150;
trainOpts.continue=false;
trainOpts.gpus=[];
trainOpts.learningRate=0.001;
trainOpts.expDir='data/test_ip_patch' ;
trainOpts.numSubBatches=1;

%getBatch options

bopts.useGpu=numel(trainOpts.gpus)>0;

%%%%network definition

net=dagnn.DagNN();
net.addLayer('conv1',dagnn.Conv('size',[2 2 200 200], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'input'}, {'conv1'}, {'conv1f'  'conv1b'});
net.addLayer('pool1', dagnn.Pooling ('method', 'max', 'poolSize', [ 2, 2],...
    'stride', [1 1 ], 'pad', [0 0 0 0 ]), {'conv1'}, {'pool1'},{});



net.addLayer('conv2',dagnn.ConvTranspose('size',[2 2 200 200], 'hasBias', 'true'),{'pool1'}, {'conv2'}, {'conv2f'  'conv2b'});
net.addLayer('relu',dagnn.ReLU(),{'conv2'},{'relu'},{});


net.addLayer('classifier',dagnn.Conv('size',[4 4 200 16], 'hasBias', 'true','stride', [1, 1],...
    'pad', [0 0 0 0]), {'relu'}, {'classifier'}, {'conv3f'  'conv3b'});
net.addLayer('prediction',dagnn.SoftMax(),{'classifier'},{'prediction'},{});
net.addLayer('objective',dagnn.Loss('loss', 'log'), {'prediction','label'},{'objective'},{});
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction', 'label'},'error');

%%%%%   end of the network   %%%%%%%%%%%%%%

%%%%%%% initialization of the weights%%%%%%%%
initNet(net, 1/100);

%%%%%%%%%%%  training %%%%%%%%%%
rng('default')
net=cnn_train_dag(net,imdb, @getBatchA,trainOpts);
end

function initNet(net,f)
net.initParams();
%%%%if is a convolution layer%%%%%%%%%
for l=1:length(net.layers)
    if(strcmp(class(net.layers(l).block),'dagnn.Conv'))
        f_ind=net.layers(l).paramIndexes(1);
        b_ind=net.layers(1).paramIndexes(2);
        
        
        net.params(f_ind).value=f*randn(size(net.params(f_ind).value), 'single');
        net.params(f_ind).learningRate=1;
        net.params(f_ind).weightDecay=1;
        
        
        net.params(b_ind).value=f*randn(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate=2;
        net.params(b_ind).weightDecay=1;

    end
    
end
end



%%%%%%%%%%%%function on charge of creating a batch of images+labels
function inputs=getBatchA(imdb, batch)
images=imdb.images.data(:,:,:,batch);
labels=imdb.images.label(1,batch);

inputs={'input',images,'label',labels};
end