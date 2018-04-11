% Copyright (c) Hu Zhiming 2018/4/4 JimmyHu@pku.edu.cn All Rights Reserved.
% preprocess the images: split the images into train, validation & test
% sets.

% load the datasplits mat.
load 'datasplits.mat'

% the path of the original images.
imgPath1 = 'Images/';
% the path of the renamed images.
imgPath2 = 'Images2/';

% rename the original images and restore them in imgPath2
% list all the images in imgPath1.
imgDir1 = dir([imgPath1, '*.jpg']);
for i = 1: length(imgDir1)
    % read all the images.
    img = imread([imgPath1 imgDir1(i).name]);
    name2=[imgPath2 num2str(str2num(imgDir1(i).name(7:10))) '.jpg'];
    imwrite(img, name2);
end

% split the train set.
trainPath = 'Train/';
for i =1 : length(trn1)
    % the name of the training images.
    name = [imgPath2 num2str(trn1(i)) '.jpg'];
    img = imread(name);
    name2 = [trainPath num2str(trn1(i)) '.jpg'];
    imwrite(img, name2);
end

% split the validation set.
testPath = 'Validation/';
for i =1 : length(val1)
    % the name of the training images.
    name = [imgPath2 num2str(val1(i)) '.jpg'];
    img = imread(name);
    name2 = [testPath num2str(val1(i)) '.jpg'];
    imwrite(img, name2);
end

% split the test set.
testPath = 'Test/';
for i =1 : length(tst1)
    % the name of the training images.
    name = [imgPath2 num2str(tst1(i)) '.jpg'];
    img = imread(name);
    name2 = [testPath num2str(tst1(i)) '.jpg'];
    imwrite(img, name2);
end

% Data Augmentation for the original train images.
trainPath = 'Train/';
% list all the images in trainPath.
trainDir = dir([trainPath, '*.jpg']);

% Randomly flip the original images.
flipPath = 'TrainFlip/';
for i = 1: length(trainDir)
    % the random flag.
    flag = rand;
    if flag > 0.5
        % read the original train images.
        img = imread([trainPath trainDir(i).name]);
        % flip the original image.
        img2 = fliplr(img);
        name2=[flipPath trainDir(i).name];
        imwrite(img2, name2);
    end
end

% Randomly crop the original images.
cropPath = 'TrainCrop/';
for i = 1: length(trainDir)
    % the random flag.
    flag = rand;
    if flag > 0.5
        % read the original train images.
        img = imread([trainPath trainDir(i).name]);
        % resize the original image.
        img2 = imresize(img, 1.2);
        width = size(img, 2);
        height = size(img, 1);
        % crop the image.
        img2 = imcrop(img2, [width*0.1 height*0.1 width height]);
        name2=[cropPath trainDir(i).name];
        imwrite(img2, name2);
    end
end

% Randomly crop the original images.
noisePath = 'TrainNoise/';
for i = 1: length(trainDir)
    % the random flag.
    flag = rand;
    if flag > 0.5
        % read the original train images.
        img = imread([trainPath trainDir(i).name]);
        % add noise to the original image.
        img2 = imnoise(img, 'salt & pepper', 0.02);
        name2=[noisePath trainDir(i).name];
        imwrite(img2, name2);
    end
end
