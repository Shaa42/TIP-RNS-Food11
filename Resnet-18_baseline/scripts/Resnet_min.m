clear;
close all;
clc;

%% 1. 定位项目、数据集和输出目录
scriptDir = fileparts(mfilename("fullpath"));
baselineDir = fileparts(scriptDir);
repoDir = fileparts(baselineDir);

resultsDir = fullfile(baselineDir,"results");
modelsDir = fullfile(baselineDir,"models");
checkpointsDir = fullfile(baselineDir,"checkpoints");

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end
if ~isfolder(modelsDir)
    mkdir(modelsDir);
end
if ~isfolder(checkpointsDir)
    mkdir(checkpointsDir);
end

trainFolder = fullfile(repoDir, ...
    "food-11-resized", "train");

assert(isfolder(trainFolder), ...
    "Training folder not found: %s",trainFolder);

%% 2. 读取有标签的训练图片
imdsAll = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

disp(countEachLabel(imdsAll));

%% 3. 按每个类别划分80%训练、20%验证
rng(42);  % 固定随机种子，保证实验可复现

[imdsTrain, imdsValidation] = splitEachLabel( ...
    imdsAll, 0.8, "randomized");

fprintf("Training images: %d\n", ...
    numel(imdsTrain.Files));

fprintf("Validation images: %d\n", ...
    numel(imdsValidation.Files));

%% 4. 加载预训练ResNet-18并修改为11类输出
numClasses = numel(categories(imdsTrain.Labels));

net = imagePretrainedNetwork("resnet18", ...
    NumClasses=numClasses);

disp(net.Layers(1).InputSize);

%% 5. 冻结ResNet-18主干，只训练最后的分类层
learnables = net.Learnables;
headLayer = string(learnables.Layer(end));

fprintf("Trainable classification head: %s\n", headLayer);

for i = 1:height(learnables)
    layerName = string(learnables.Layer(i));
    parameterName = string(learnables.Parameter(i));

    if layerName ~= headLayer
        net = setLearnRateFactor( ...
            net, layerName, parameterName, 0);
    end
end

%% 6. 按类别数量计算加权交叉熵
classNames = categories(imdsTrain.Labels);
classCounts = countcats(imdsTrain.Labels);

classWeights = sum(classCounts) ./ ...
    (numClasses * classCounts);
% crossentropy expects class weights as a 1-by-C row vector.
classWeights = single(classWeights(:)');

disp(table(string(classNames),classCounts,classWeights', ...
    VariableNames=["Class","TrainingImages","Weight"]));

lossFcn = @(Y,T) crossentropy( ...
    Y,T,classWeights,WeightsFormat="UC");

%% 7. 设置快速训练参数
miniBatchSize = 64;

iterationsPerEpoch = ceil( ...
    numel(imdsTrain.Files)/miniBatchSize);

fprintf("Iterations per epoch: %d\n",iterationsPerEpoch);
fprintf("Maximum training iterations: %d\n",3*iterationsPerEpoch);

checkpointFolder = fullfile(checkpointsDir,"resnet18_head");
if ~isfolder(checkpointFolder)
    mkdir(checkpointFolder);
end

options = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=3, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=max(1,iterationsPerEpoch), ...
    ValidationPatience=2, ...
    Metrics="accuracy", ...
    OutputNetwork="best-validation", ...
    ExecutionEnvironment="auto", ...
    CheckpointPath=checkpointFolder, ...
    CheckpointFrequency=1, ...
    Plots="training-progress", ...
    Verbose=true, ...
    VerboseFrequency=20);

%% 8. 迁移学习：仅更新分类头
[netTrained,trainingInfo] = trainnet( ...
    imdsTrain, ...
    net, ...
    lossFcn, ...
    options);

%% 9. 验证集预测
validationScores = minibatchpredict( ...
    netTrained, imdsValidation);

validationPredictions = scores2label( ...
    validationScores, classNames);

validationAccuracy = mean( ...
    validationPredictions == imdsValidation.Labels);

fprintf("Validation accuracy: %.2f%%\n", ...
    100*validationAccuracy);

%% 10. 计算每类F1和Macro-F1
classOrder = categorical(classNames,classNames);
confusionMatrix = confusionmat( ...
    imdsValidation.Labels,validationPredictions, ...
    Order=classOrder);

truePositive = diag(confusionMatrix);
precision = truePositive ./ max(sum(confusionMatrix,1)',1);
recall = truePositive ./ max(sum(confusionMatrix,2),1);
f1PerClass = 2 .* precision .* recall ./ ...
    max(precision + recall,eps);
macroF1 = mean(f1PerClass);

disp(table(string(classNames),precision,recall,f1PerClass, ...
    VariableNames=["Class","Precision","Recall","F1"]));
fprintf("Validation macro-F1: %.4f\n",macroF1);

%% 11. 归一化混淆矩阵
figure;
cm = confusionchart( ...
    imdsValidation.Labels, ...
    validationPredictions);
cm.Normalization = "row-normalized";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";
title("ResNet-18 Head-Only Validation Confusion Matrix");
exportgraphics(gcf,fullfile(resultsDir, ...
    "confusion_matrix_head_only.png"),Resolution=200);

%% 12. 保存模型和训练信息
save(fullfile(modelsDir, ...
    "food11_resnet18_head_only.mat"), ...
    "netTrained", ...
    "classNames", ...
    "validationAccuracy", ...
    "macroF1", ...
    "f1PerClass", ...
    "trainingInfo");

%% 13. 读取无标签测试集
testFolder = fullfile(repoDir, ...
    "food-11-resized", "test");

assert(isfolder(testFolder), ...
    "Test folder not found: %s",testFolder);

imdsTest = imageDatastore(testFolder);

fprintf("Test images: %d\n", ...
    numel(imdsTest.Files));

assert(numel(imdsTest.Files) == 3097, ...
    "Test set should contain exactly 3097 images.");

%% 14. 对测试集预测
testScores = minibatchpredict( ...
    netTrained, imdsTest);

testPredictions = scores2label( ...
    testScores, classNames);

%% 15. 提取不带扩展名的图片编号
testFiles = string(imdsTest.Files);
testIds = strings(numel(testFiles),1);

for i = 1:numel(testFiles)
    [~,testIds(i),~] = fileparts(testFiles(i));
end

assert(numel(unique(testIds)) == numel(testIds), ...
    "Duplicate test image identifiers detected.");

%% 16. 生成JSON映射
jsonKeys = cellstr(testIds);
jsonValues = cellstr(string(testPredictions));

predictionMap = containers.Map( ...
    jsonKeys, jsonValues);

jsonText = jsonencode( ...
    predictionMap, PrettyPrint=true);

%% 17. 保存JSON文件
jsonPath = fullfile(resultsDir, ...
    "food11_resnet18_head_only_predictions.json");

fileId = fopen(jsonPath,"w");

assert(fileId ~= -1, ...
    "Could not create JSON file.");

fprintf(fileId,"%s",jsonText);
fclose(fileId);

fprintf("JSON saved to:\n%s\n",jsonPath);
