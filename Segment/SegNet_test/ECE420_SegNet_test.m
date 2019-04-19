%% Test the Network
%% Classify Test Data
load net.mat

imageDir       = fullfile('imdsTest'); % Location of images
labelDir       = fullfile('pxdsTest'); % Location of labels 

%% Define class names and associated label IDs
classNames      = ["table", "skin","paper","keyboard","mouse","monitor","background"];
labelIDs        = [43 85 128 170 212 255 0];

XTest = imageDatastore(imageDir);
YTest = pixelLabelDatastore(labelDir,classNames,labelIDs);


num_test      = size(XTest.Files,1);
for i = 1:num_test
I             = imread(XTest.Files{i});
C             = semanticseg(I,net);
I_test        = zeros([1080,1920]); % initialize vector to add segmented overlay to image
I_table       = 1*(C=="table"); % change the numbers for each label
I_skin        = 2*(C=="skin");
I_paper       = 3*(C=="paper");
I_keyboard    = 4*(C=="keyboard");
I_mouse       = 5*(C=="mouse");
I_monitor     = 6*(C=="monitor");
I_background  = 7*(C=="background");
I_test        = I_test + I_table + I_skin + I_paper + I_keyboard + I_mouse + I_monitor + I_background;
I_bin         = 255*(C == 'c'); 
[filepath,name,ext] = fileparts(XTest.Files{i});
     imwrite(I_bin, "testing/"+name+".png");
end

%% Calculate Classification Accuracy of Predictions

YPred = semanticseg(XTest,net);

metrics = evaluateSemanticSegmentation(YPred,YTest);
metrics.ClassMetrics
metrics.ConfusionMatrix

% Show confusion matrix
normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames,classNames,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';


% Add CamVid pixel labels -> https://www.mathworks.com/help/vision/ug/semantic-segmentation-examples.html#mw_9ca2a7be-c8c2-4bbf-b168-128261d1be7d