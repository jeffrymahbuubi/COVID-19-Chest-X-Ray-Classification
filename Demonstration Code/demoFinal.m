%================================================================================================
% Function: Demo how the network classify image and the corresponding Score
% CAM
% Author: Aunuun Jeffry Mahbuubi / 馬杰睿 /F04087189
% Date: 2022/01/22
%================================================================================================

%% Image Processing
clc;
%Original Image
[FileName, PathName] = uigetfile({'*.jpg;*.tif;*.png;*.gif','All Image Files'},'Please select an Image');
Img = imread([PathName FileName]);
dirImg = fullfile([PathName FileName]);
figure(1);
imshow(Img);
title('Original Lungs Image');

%Histogram Equalization Image
ImHE = histeq(Img);
figure(2);
imshow(ImHE);
title('Histogram Equalization Lungs Image');

%Gamma Correction Image
ImGamma = lin2rgb(Img);
figure(3);
imshow(ImGamma);
title('Gamma Correction Lungs Image');

% Segmented Image
load('unet_retrain2.mat')
[C, scores] = semanticseg(Img,net);
BW = C == 'lungs';
ImSegmented = zeros(size(Img),'like',Img);
ImSegmented(~BW) = Img(~BW);
figure(4);
imshow(ImSegmented);
title('Segmented Lungs Image');

%% Classification 
clc;
%ResNet18 Input Size
inputSize = [224 224 3];

% Original 
load('ResNet18_Original.mat');
augimdsTesting = augmentedImageDatastore(inputSize(1:2),Img,'ColorPreprocessing','gray2rgb');
[labels,score] = classify(Modelori, augimdsTesting);
fprintf('The classification Original class is: %s\n',labels);

map_o = gradCAM(Modelori,editimagever2(Img),labels);
figure(5)
imshow(editimagever2(Img),'InitialMagnification','fit')
hold on
ho = imagesc(map_o,'AlphaData',0.3);
colormap jet
hold off
title('Original Image Lungs Score-CAM ');


%-----------------------------------------------------------------------------------------------%

%Histogram Equalization
load('ResNet18_Histogram.mat');
augimdsTesting = augmentedImageDatastore(inputSize(1:2),ImHE,'ColorPreprocessing','gray2rgb');
[labels,score] = classify(Modelhe, augimdsTesting);
fprintf('The classification Histogram Equalizatoin class is: %s\n',labels);

map_h = gradCAM(Modelhe,editimagever2(ImHE),labels);
figure(6)
imshow(editimagever2(ImHE),'InitialMagnification','fit')
hold on
hh = imagesc(map_h,'AlphaData',0.3);
colormap jet
hold off
title('Histogram Equalization Lungs Image Score-CAM ');


%-----------------------------------------------------------------------------------------------%

%Gamma Correction
load('ResNet18_Gamma.mat');
augimdsTesting = augmentedImageDatastore(inputSize(1:2),ImGamma,'ColorPreprocessing','gray2rgb');
[labels,score] = classify(Modelgamma, augimdsTesting);
fprintf('The classification Gamma Correction class is: %s\n',labels);

map_g = gradCAM(Modelgamma,editimagever2(ImGamma),labels);
figure(7)
imshow(editimagever2(ImGamma),'InitialMagnification','fit')
hold on
hg = imagesc(map_g,'AlphaData',0.3);
colormap jet
hold off
title('Gamma Correction Lungs Image Score-CAM ');

%-----------------------------------------------------------------------------------------------%

%Segmented Image
load('ResNet18_Segmented.mat');
augimdsTesting = augmentedImageDatastore(inputSize(1:2),ImSegmented,'ColorPreprocessing','gray2rgb');
[labels,score] = classify(Modelsegmented, augimdsTesting);
fprintf('The classification Image Segmentation class is: %s\n',labels);

map = gradCAM(Modelsegmented,editimagever2(Img),labels);
I = editimage(ImSegmented);

figure(8)
imshow(editimagever2(Img),'InitialMagnification','fit')
hold on
h = imagesc(map,'AlphaData',0.3);
colormap jet
hold off
set(h,'AlphaData',I)
title('Segmented Lungs Image Score-CAM ');