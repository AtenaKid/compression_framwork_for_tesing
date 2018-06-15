% run('matconvnet-1.0-beta23\matlab\vl_setupnn.m') ;
addpath('matconvnet-1.0-beta23\matlab\mex'); 

% clear; clc;
addpath('utilities');
%folderTest  = 'Train400';

showResult  = 1;
useGPU      = 1;
pauseTime   = 1;

JPEG_Quality = 50;   
net1.layers = {};
net2.layers = {};
%load ComCNN
load(fullfile('model','ComCNN_QF=50.mat'));
net.layers = net.layers(1:end-1); 
net1 = vl_simplenn_tidy(net);

%load RecCNN
load(fullfile('model','MWCNN_Haart_SRx2.mat'));
net2 = dagnn.DagNN.loadobj(net);
net2.removeLayer('objective') ;
out_idx = net.getVarIndex('prediction') ;
net2.vars(net.getVarIndex('prediction')).precious = 1 ;
net2.mode = 'test';

%%% move to gpu
if useGPU
    net1 = vl_simplenn_move(net1, 'gpu') ;
    net2.move('gpu') ;
end

%read image
% label = imread('butterfly.bmp'); 
label = imread('image/Lena.png');

if size(label,3)>1
    label = rgb2gray(label);
end

label = im2single(label);

[hei,wid] = size(label);
if useGPU
    label = gpuArray(label);
end

% tic
res = vl_simplenn(net1,label,[],[],'conserveMemory',true,'mode','test');
Low_Resolution = res(end).x;
% toc

if useGPU
    Low_Resolution = gather(Low_Resolution);
end

imwrite(im2uint8(Low_Resolution),'Results/compressed_image.jpg','jpg','Quality',JPEG_Quality);%Compression

im_input = im2single(imread('Results/compressed_image.jpg'));
input = imresize(im_input,[hei,wid],'bicubic');

%%% convert to GPU
if useGPU
    input = gpuArray(input);
end
% tic
%res    = vl_simplenn(net2,input,[],[],'conserveMemory',true,'mode','test');

output = input - res(end).x;
% toc
[PSNRWithNet, SSIMWithNet] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);

if useGPU
    output = gather(output);
    input  = gather(input);
end
imwrite(im2uint8(Low_Resolution),'Results/output.jpg','jpg')
figure, imshow(label); title(sprintf('Raw-Input'));

figure, imshow(output); title(sprintf('After CNN Network, PSNR: %.2f dB,SSIM: %.4f', PSNRWithNet, SSIMWithNet));

if useGPU
    label = gather(label);
end
JPEG_Quality1= 10;
imwrite(label,'Results/JPEG-Directly.jpg','jpg','Quality',JPEG_Quality1);%
im_direct = im2single(imread('Results/JPEG-Directly.jpg'));
[PSNR_direct, SSIM_direct] = Cal_PSNRSSIM(im2uint8(label),im2uint8(im_direct),0,0);
figure, imshow(im_direct); title(sprintf('JPEG-Directly, PSNR: %.2f dB,SSIM: %.4f', PSNR_direct, SSIM_direct));



