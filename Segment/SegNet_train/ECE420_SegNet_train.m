%%  SegNet %%%%% 
clear all;
close all;
clc;


%% Load data %%%%%
% Image directories
% imageDir       = fullfile('colorImages'); % Location of images
% labelDir       = fullfile('grayscaleImages'); % Location of labels 
% 
% %% Create image data store holding the training images
% imds = imageDatastore(imageDir);
XTrain = imageDatastore("./Train/imageDir"); 
XTest = imageDatastore("./Test/imageDir");
%% Define class names and associated label IDs
classNames      = ["table", "skin","paper","keyboard","mouse","monitor","background"];
labelIDs        = [43 85 128 170 212 255 0];

%% Create a pixel label datastore holding the ground truth pixel labels for the training images.
%pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
YTrain = pixelLabelDatastore("./Train/labelDir",classNames,labelIDs);
YTest = pixelLabelDatastore("./Test/labelDir",classNames,labelIDs);

%% Create SegNet layers.
imageSize       = [1080 1920 3]; % [height width depth] of image -> choose depth of 3 for RGB image and depth of 1 for gray image
numClasses      = 7;
encoderDepth    = 3;
lgraph          = segnetLayers(imageSize,numClasses,encoderDepth);
%lgraph = segnetLayers(imageSize,numClasses,'vgg16'); %Use pretrained network vgg16 -> encoderDepth = 5 for this network

%% Display the Network
% plot(lgraph)
% title('Network Layer Graph')



%% Create a pixel label image datastore for training 
pximds          = pixelLabelImageDatastore(XTrain,YTrain);                                

%% Define Network Options %%%%%
%numClasses = 7; % For Fully Connected Layer -> 7 classes -> 0:background 1:table(43), 2:skin(85), 3:paper(128), 4:keyboard(170), 5:mouse(212), 6:monitor(255)
maxEpochs = 150;
miniBatchSize = 2;
options = trainingOptions('sgdm','InitialLearnRate',0.01, ...
    'MiniBatchSize',miniBatchSize, ...
      'MaxEpochs',maxEpochs, ...
      'Verbose',true, ...
      'VerboseFrequency',25);
  
  fprintf('Encoder Depth: %d \n', encoderDepth);
  fprintf('Maximum Number of Epochs: %d \n', maxEpochs);
  fprintf('Mini Batch Size: %d \n', miniBatchSize);

%% Train the Network
net             = trainNetwork(pximds,lgraph,options);

save('net','net');

