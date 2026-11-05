clear;
close all;
clc;

%% 1. Project and data paths
scriptPath = mfilename("fullpath");
projectDir = fileparts(scriptPath);
trainFolder = fullfile(projectDir,"food-11-resized","train");
testFolder = fullfile(projectDir,"food-11-resized","test");

assert(isfolder(trainFolder),"Training folder not found: %s",trainFolder);
assert(isfolder(testFolder),"Test folder not found: %s",testFolder);

%% 2. Recreate the same reproducible 80/20 split
imdsAll = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

rng(42);
[imdsTrain,imdsValidation] = splitEachLabel( ...
    imdsAll,0.8,"randomized");

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

fprintf("Training images: %d\n",numel(imdsTrain.Files));
fprintf("Validation images: %d\n",numel(imdsValidation.Files));
disp(countEachLabel(imdsTrain));

%% 3. Start independently from ImageNet-pretrained ResNet-18
net = imagePretrainedNetwork("resnet18",NumClasses=numClasses);

%% 4. Moderate class weighting
% Square-root inverse-frequency weighting is less aggressive than the
% head-only experiment and should reduce excessive false positives.
classCounts = countcats(imdsTrain.Labels);
classWeights = sqrt(sum(classCounts)./(numClasses*classCounts));
classWeights = classWeights/mean(classWeights);
classWeights = single(classWeights(:)');

disp(table(string(classNames),classCounts,classWeights', ...
    VariableNames=["Class","TrainingImages","Weight"]));

lossFcn = @(Y,T) crossentropy( ...
    Y,T,classWeights,WeightsFormat="UC");

%% 5. Lightweight data augmentation
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-10 10], ...
    RandXTranslation=[-8 8], ...
    RandYTranslation=[-8 8], ...
    RandScale=[0.9 1.1]);

augmentedTrain = augmentedImageDatastore( ...
    [224 224],imdsTrain,DataAugmentation=augmenter);

%% 6. Shared training configuration
miniBatchSize = 64;
warmupEpochs = 3;
fineTuneEpochs = 8;
iterationsPerEpoch = ceil(numel(imdsTrain.Files)/miniBatchSize);

fprintf("Iterations per epoch: %d\n",iterationsPerEpoch);
fprintf("Stage 1 maximum iterations: %d\n", ...
    warmupEpochs*iterationsPerEpoch);
fprintf("Stage 2 maximum iterations: %d\n", ...
    fineTuneEpochs*iterationsPerEpoch);

warmupCheckpointFolder = fullfile( ...
    projectDir,"checkpoints_resnet18_full_stage1");
fineTuneCheckpointFolder = fullfile( ...
    projectDir,"checkpoints_resnet18_full_stage2");

if ~isfolder(warmupCheckpointFolder)
    mkdir(warmupCheckpointFolder);
end
if ~isfolder(fineTuneCheckpointFolder)
    mkdir(fineTuneCheckpointFolder);
end

%% 7. Stage 1: freeze the backbone and train the classification head
learnables = net.Learnables;
headLayer = string(learnables.Layer(end));

for i = 1:height(learnables)
    layerName = string(learnables.Layer(i));
    parameterName = string(learnables.Parameter(i));
    factor = double(layerName == headLayer);
    net = setLearnRateFactor(net,layerName,parameterName,factor);
end

fprintf("Stage 1 - trainable classification head: %s\n",headLayer);

warmupOptions = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=warmupEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=max(1,iterationsPerEpoch), ...
    Metrics="accuracy", ...
    OutputNetwork="best-validation", ...
    ExecutionEnvironment="auto", ...
    CheckpointPath=warmupCheckpointFolder, ...
    CheckpointFrequency=1, ...
    Plots="training-progress", ...
    Verbose=true, ...
    VerboseFrequency=20);

[netWarmup,warmupInfo] = trainnet( ...
    augmentedTrain,net,lossFcn,warmupOptions);

%% 8. Stage 2: unfreeze the final ResNet stage and fine-tune
learnables = netWarmup.Learnables;
learnableLayerNames = unique(string(learnables.Layer),"stable");
numFineTuneLayers = min(11,numel(learnableLayerNames));
fineTuneLayers = learnableLayerNames(end-numFineTuneLayers+1:end);

for i = 1:height(learnables)
    layerName = string(learnables.Layer(i));
    parameterName = string(learnables.Parameter(i));
    factor = double(ismember(layerName,fineTuneLayers));
    netWarmup = setLearnRateFactor( ...
        netWarmup,layerName,parameterName,factor);
end

fprintf("Stage 2 - fine-tuning the final %d learnable layers:\n", ...
    numFineTuneLayers);
disp(fineTuneLayers);

reset(augmentedTrain);

fineTuneOptions = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=fineTuneEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=max(1,iterationsPerEpoch), ...
    ValidationPatience=3, ...
    Metrics="accuracy", ...
    OutputNetwork="best-validation", ...
    ExecutionEnvironment="auto", ...
    CheckpointPath=fineTuneCheckpointFolder, ...
    CheckpointFrequency=1, ...
    Plots="training-progress", ...
    Verbose=true, ...
    VerboseFrequency=20);

[netTrained,fineTuneInfo] = trainnet( ...
    augmentedTrain,netWarmup,lossFcn,fineTuneOptions);

%% 9. Validation metrics
validationScores = minibatchpredict(netTrained,imdsValidation);
validationPredictions = scores2label(validationScores,classNames);
validationAccuracy = mean(validationPredictions == imdsValidation.Labels);

classOrder = categorical(classNames,classNames);
confusionMatrix = confusionmat( ...
    imdsValidation.Labels,validationPredictions,Order=classOrder);

truePositive = diag(confusionMatrix);
precision = truePositive./max(sum(confusionMatrix,1)',1);
recall = truePositive./max(sum(confusionMatrix,2),1);
f1PerClass = 2.*precision.*recall./max(precision+recall,eps);
macroF1 = mean(f1PerClass);

fprintf("Validation accuracy: %.2f%%\n",100*validationAccuracy);
fprintf("Validation macro-F1: %.4f\n",macroF1);
disp(table(string(classNames),precision,recall,f1PerClass, ...
    VariableNames=["Class","Precision","Recall","F1"]));

figure;
cm = confusionchart(imdsValidation.Labels,validationPredictions);
cm.Normalization = "row-normalized";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";
title("ResNet-18 Fine-Tuned Validation Confusion Matrix");

%% 10. Save the complete model
modelPath = fullfile(projectDir,"food11_resnet18_full.mat");
save(modelPath, ...
    "netTrained","classNames","validationAccuracy","macroF1", ...
    "f1PerClass","precision","recall", ...
    "warmupInfo","fineTuneInfo","-v7.3");
fprintf("Model saved to:\n%s\n",modelPath);

%% 11. Test prediction and Moodle JSON
imdsTest = imageDatastore(testFolder);
assert(numel(imdsTest.Files) == 3097, ...
    "Test set should contain exactly 3097 images.");

testScores = minibatchpredict(netTrained,imdsTest);
testPredictions = scores2label(testScores,classNames);

testFiles = string(imdsTest.Files);
testIds = strings(numel(testFiles),1);
for i = 1:numel(testFiles)
    [~,testIds(i),~] = fileparts(testFiles(i));
end

assert(numel(unique(testIds)) == numel(testIds), ...
    "Duplicate test image identifiers detected.");

predictionMap = containers.Map( ...
    cellstr(testIds),cellstr(string(testPredictions)));
jsonText = jsonencode(predictionMap,PrettyPrint=true);
jsonPath = fullfile(projectDir, ...
    "food11_resnet18_full_predictions.json");

fileId = fopen(jsonPath,"w");
assert(fileId ~= -1,"Could not create JSON file.");
cleanupFile = onCleanup(@() fclose(fileId));
fprintf(fileId,"%s",jsonText);
clear cleanupFile;

fprintf("Test images: %d\n",numel(imdsTest.Files));
fprintf("JSON saved to:\n%s\n",jsonPath);
