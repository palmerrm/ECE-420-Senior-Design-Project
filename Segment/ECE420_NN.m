%%%%%  Neural Network %%%%% 
clear all;
close all;
clc;


%% Load data %%%%%
% Image directories
imageDir       = fullfile('colorImages'); % Location of images
labelDir       = fullfile('grayscaleImages'); % Location of labels 

%Create image data store holding the training images
imds = imageDatastore(imageDir);
XTrain = imageDatastore("./Train/imageDir");
XTest = imageDatastore("./Test/imageDir");
% Define class names and associated label IDs
classNames      = ["table", "skin","paper","keyboard","mouse","monitor","background"];
labelIDs        = [43 85 128 170 212 255 0];

%Create a pixel label datastore holding the ground truth pixel labels for the training images.
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
YTrain = pixelLabelDatastore("./Train/labelDir",classNames,labelIDs);
YTest = pixelLabelDatastore("./Test/labelDir",classNames,labelIDs);

%%Create SegNet layers.
imageSize       = [1080 1920 3]; % [height width depth] of image -> choose depth of 3 for RGB image and depth of 1 for gray image
numClasses      = 7;
encoderDepth    = 5;
lgraph          = segnetLayers(imageSize,numClasses,encoderDepth);
%lgraph = segnetLayers(imageSize,numClasses,'vgg16'); %Use pretrained network vgg16 -> encoderDepth = 5 for this network
                                

%%Create a pixel label image datastore for training a semantic segmentation network.
pximds          = pixelLabelImageDatastore(XTrain,YTrain);                                



%% Define Network Options %%%%%
%numClasses = 7; % For Fully Connected Layer -> 7 classes -> 0:background 1:table(43), 2:skin(85), 3:paper(128), 4:keyboard(170), 5:mouse(212), 6:monitor(255)
maxEpochs = 100;
miniBatchSize = 28;
options = trainingOptions('sgdm','InitialLearnRate',1e-3, ...
      'MaxEpochs',20,'VerboseFrequency',10);

%% Train the Network
net             = trainNetwork(pximds,lgraph,options);

%% Display the Network
plot(lgraph)


% % %% Test the Network
% % [XTest,YTest] = testData;
% % XTest(1:3:)
% % 
% % % Classify Test Data
% % miniBatchSize = 27; % Mini batch-size to divide amount of training data evenly and reduce amount of padding in mini-batches
% % YPred = classify(net,XTest, ...
% %     'MiniBatchSize',miniBatchSize); %reduce the amount of padding introduced by the classification process, set the mini-batch size to 27
% %     
% % % Calculate Classification Accuracy of Predictions
% % accuracy = sum(YPred == YTest)./numel(YTest);