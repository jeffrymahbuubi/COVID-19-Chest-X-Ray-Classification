%================================================================================================
% Function: Training the deep learning model
% Input: "Model"  --> The name of CNN Network we choose
%        "imdsTrain"   --> Data Store of Training Data
%        "imdsValidation"--> Data Store of Testing Data
%================================================================================================

function  [Mdl, labels, score] = DL_trainResNet18(Model, imdsTrain, imdsValidation)

if strcmp(Model, 'ResNet18')
    net = resnet18;
    
    layersTransfer = net.Layers(1:end-3);
    numClasses = numel(categories(imdsTrain.Labels));

    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'Name','fc1000')
        softmaxLayer('Name','prob')
        classificationLayer('Name','ClassificationLayer_predictions')];

    connections = net.Connections;
    layers = createLgraphUsingConnections(layers,connections);

    inputSize = net.Layers(1).InputSize;
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    
    if (size(imread(imdsTrain.Files{1}),3)==1)          % check RGB channel
        augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing','gray2rgb');
    else
        augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    end
    
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ...
        'MaxEpochs',20, ...
        'InitialLearnRate',0.001, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',3, ...
        'Verbose',false,...
        'Momentum',0.9,...
        'Plots','training-progress');

    Mdl = trainNetwork(augimdsTrain,layers,options);
    [labels,score] = classify(Mdl, augimdsValidation);
end
end