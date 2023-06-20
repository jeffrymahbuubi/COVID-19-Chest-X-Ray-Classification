%================================================================================================
% Function: Training the Segmentation Network
% Author: Aunuun Jeffry Mahbuubi / 馬杰睿 / F04087189
% Date: 2022/01/22
%================================================================================================

%% Code to Do The Training of UNet
clc;

dataFolder = fullfile(['C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\' ...
    'Computer Aided Engineering\Quiz, Mid-Term, Final-Project\Final Project\Group 1\Dataset\Image Segmentation Dataset']);
imageDir = fullfile(dataFolder,'xray');
labelDir = fullfile(dataFolder,'mask');

%Create an imageDatastore for the images.
imdsTrain = imageDatastore(imageDir,'IncludeSubfolders',true);

%Create a pixelLabelDatastore for the ground truth pixel labels. 
classNames = ["lungs", "background"];
labels = [0 1];
pxdsTrain = pixelLabelDatastore(labelDir,classNames,labels,'IncludeSubfolders',true);

ds = pixelLabelImageDatastore(imdsTrain,pxdsTrain); %Combine Image data with mask data for training

tbl = countEachLabel(pxdsTrain);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
inverseFrequency = 1./frequency;

layerf = pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',inverseFrequency);

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'LearnRateDropFactor',5e-1, ...
    'LearnRateDropPeriod',20, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',4, ...
    'Plots','training-progress');

imageSize = [256 256]; %Specify the image size of the input data.
numClasses = 2; %Number of classes in image
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',4);

net = trainNetwork(ds,lgraph,options);
% save('unet_retrain4.mat','net');

%% Code to Extract the Lung Region after finished UNet Training
load('unet_retrain2.mat') %Load the trained Segmentation Network

ImPath = ['C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering' ...
    '\Quiz, Mid-Term, Final-Project\Final Project\Deep Learning\train\COVID\'];
ext = {'*.png'};
imlist = [];
for i = 1:length(ext)
    list = dir([ImPath ext{i}]);
    imlist = [imlist; list];
end

dataFolder = fullfile(['C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering' ...
    '\Quiz, Mid-Term, Final-Project\Final Project\Deep Learning\train']);
folder1 = fullfile(dataFolder,'COVID'); 
folderout = fullfile(dataFolder,'Extract COVID'); % Location of the Image Output
filelist1 = dir(fullfile(folder1, '*.png'));
filelist1 = natsortfiles({filelist1.name});

%Below is to combine the generated mask with X-Ray Images, the result would
%be the extracted lung region.
for i = 1:length(imlist)
    outfilepattern = string(filelist1(i));
    I = imread(fullfile(folder1, filelist1{i}));
    I = imresize(I,[256 256]);
    [C, scores] = semanticseg(I,net);
    
    BW = C == 'lungs';
    
    Aseg2 = zeros(size(I),'like',I);
    Aseg2(~BW) = I(~BW);

    %% save the mask
    imwrite(Aseg2, fullfile(folderout, sprintf(outfilepattern, i)));
end