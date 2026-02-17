clear;clc;

% "http://marsyas.info/downloads/datasets.html" dataset is available in this given website

notefolder=fullfile('genres','blues');
listname=dir(fullfile(notefolder,'*.wav'));
coeffs1=zeros(length(listname),186);

for k=1:length(listname)
    file_name=fullfile(notefolder,listname(k).name);
    [x, fs]=audioread(file_name);
    
    aFE=audioFeatureExtractor('SpectralDescriptorInput',"melSpectrum",'SampleRate',fs,"mfcc",true,"spectralCentroid",true,"spectralRolloffPoint",true);

    
    z=ZCR(x);%zero crossing rate
    rms=sqrt(mean(x.^2, 1));%rmse value
    features=extract(aFE,x);%mfcc and spectral features
    kovaryans=cov(features(:,1:13));
    coeffs1(k,:)=[mean(features,1) reshape(kovaryans,1,[]) z rms];
   
end
%%
notefolder=fullfile('genres','metal');
listname=dir(fullfile(notefolder,'*.wav'));
coeffs2=zeros(length(listname),186);

for k=1:length(listname)
    file_name=fullfile(notefolder,listname(k).name);
    [x, fs]=audioread(file_name);

    aFE=audioFeatureExtractor('SpectralDescriptorInput',"melSpectrum",'SampleRate',fs,"mfcc",true,"spectralCentroid",true,"spectralRolloffPoint",true);

    
    z=ZCR(x);%zero crossing rate
    rms=sqrt(mean(x.^2, 1));%rmse value
    features=extract(aFE,x);%mfcc and spectral features
    kovaryans=cov(features(:,1:13));
    coeffs2(k,:)=[mean(features,1) reshape(kovaryans,1,[]) z rms];
    
end
%%
notefolder=fullfile('genres','country');
listname=dir(fullfile(notefolder,'*.wav'));
coeffs3=zeros(length(listname),186);

for k=1:length(listname)
    file_name=fullfile(notefolder,listname(k).name);
    [x, fs]=audioread(file_name);

    aFE=audioFeatureExtractor('SpectralDescriptorInput',"melSpectrum",'SampleRate',fs,"mfcc",true,"spectralCentroid",true,"spectralRolloffPoint",true);

    
    z=ZCR(x);%zero crossing rate
    rms=sqrt(mean(x.^2, 1));%rmse value
    features=extract(aFE,x);%mfcc and spectral features
    kovaryans=cov(features(:,1:13));
    coeffs3(k,:)=[mean(features,1) reshape(kovaryans,1,[]) z rms];
    
end
%%
notefolder=fullfile('genres','pop');
listname=dir(fullfile(notefolder,'*.wav'));
coeffs4=zeros(length(listname),186);

for k=1:length(listname)
    file_name=fullfile(notefolder,listname(k).name);
    [x, fs]=audioread(file_name);

    aFE=audioFeatureExtractor('SpectralDescriptorInput',"melSpectrum",'SampleRate',fs,"mfcc",true,"spectralCentroid",true,"spectralRolloffPoint",true);

    
    z=ZCR(x);%zero crossing rate
    rms=sqrt(mean(x.^2, 1));%rmse value
    features=extract(aFE,x);%mfcc and spectral features
    kovaryans=cov(features(:,1:13));
    coeffs4(k,:)=[mean(features,1) reshape(kovaryans,1,[]) z rms];
    
end
%%
X=[coeffs1;coeffs2;coeffs3;coeffs4]; %combining feature vectors
y=[ones(size(coeffs1,1),1); ...
   2*ones(size(coeffs2,1),1); ...
   3*ones(size(coeffs3,1),1); ...
   4*ones(size(coeffs4,1),1)]; %labeling

%split into train and test
sample_count = size(X,1);
train_count = floor(0.9 * sample_count);
indices = randperm(sample_count);

Xtrain = X(indices(1:train_count),:);
Xtest = X(indices(train_count+1:end),:);
ytrain = y(indices(1:train_count),:);
ytest = y(indices(train_count+1:end),:);

%%
%OnevsAll Logistic Regression with Regularization

num_labels=4;
lambda1=1;%for regularization

[all_theta] = oneVsAll(Xtrain, ytrain, num_labels, lambda1);
%iterasyon says deitike sonu deiiyor

predLR = predictOneVsAll(all_theta, Xtest);
fprintf('\nTest Accuracy: %f\n', mean(double(predLR == ytest)) * 100);

C2=confusionmat(ytest,predLR);
cm2=confusionchart(C2);
cm2.Title='Logistic Regression';



%%
%Neural Networks

input_layer_size=186;
hidden_layer_size=20;
num_labels=4;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 20000);%we can try different iteration number
lambda = 1;%we can try different lambda values

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, lambda);
                               
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


predNN=predict(Theta1,Theta2,Xtest);
fprintf('\nTest Accuracy: %f\n', mean(double(predNN == ytest)) * 100);


C1=confusionmat(ytest,predNN);
cm=confusionchart(C1);
cm.Title='Neural Network';
