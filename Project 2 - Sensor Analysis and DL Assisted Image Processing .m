% This MATLAB script aims at using the following toolboxes to capture
% images using a webcam connected to the computer, yield some camera
% analysis parameters such as dynamic range and SNR as well as composing a
% HDR picture from shots of the same scene under different lighting
% conditions. The image is the blurred, added noise to and lowered in
% resolution to them be processed by pre-trained deep learning networks to
% reduce noise, increase resolution and change styles. 
% 
% Need to install Image Acquisition Toolbox, Image Processing Toolbox,
% Computer Vision Toolbox, Deep learning Toolbox, ESRGAN Single Image Super
% Resolution Matlab Port

% Choose a camera from available webcams to capture images
camList = webcamlist;
camera = webcam('USB Camera VID:1133 PID:2075'); %chosen camera

% Preview the webcam image 
preview(camera);
%capture and display
img_potato=snapshot(camera); %used a potato as test subject
imshow(img_potato);
closePreview(camera);

%Image-preprocessing: Convert image to grayscale
grayImage = rgb2gray(img_potato);
imshow(grayImage);
% use image for dynamic range test: 
% source:https://www.photrio.com/forum/attachments/digitalsteptablet-jpg.274936/

%Calculate dynamic range
pixelRange = range(double(grayImage(:)));
fprintf('Dynamic Range: %f\n', pixelRange);

% Simple SNR calculation assuming uniform lighting
signal = mean(double(grayImage(:)));
noise = std(double(grayImage(:)));
snr = 10 * log10(signal / noise);
fprintf('SNR: %f dB\n', snr);

% Specify the directory and base filename
saveDir = '.../MATLAB/Projects/Project3';  % Ensure this directory exists


%Capture 10 snapshots at regular interval to allow lighting conditions to change
baseFileName = 'Snapshot_'; 
numSnapshots = 10;
interval = 3;
%Loop to capture and save images
for i = 1:numSnapshots
    img = snapshot(camera);
    fileName = sprintf('%s%s%d.jpg', saveDir, baseFileName, i);
    imwrite(img, fileName);
    fprintf('Saved %s\n', fileName);
    pause(interval);
end
clear('camera');
%Results saved in "Snapshots" in "Project 2 - Demo"

% Loop to read the images and store them in a cell array
fileExtension = '.jpg';
numImages = 10;
images = cell(1, numImages); % Initialize a cell array for the images
exposure=[]; % Initialize a cell array for exposure

for i = 1:numImages
    fileName = sprintf('%s%s%d%s', saveDir, baseFileName, i, fileExtension);
    images{i} = imread(fileName);
    %In the case of unknown exposure, use the average brightness of the Value
    % channel of the HSV color space to be the exposure value. This can work 
    % well especially since relative exposure is what matters here.
    imageHSV = rgb2hsv(images{i}); %Convert the image to HSV
    vChannel = imageHSV(:,:,3); %extract V channel
    avgBrightness= mean(vChannel(:)) * 255; %average
    % brightness calculation on scale of 0-255 
    exposure(i)=avgBrightness; %store 'exposure' value
end

% Compose an HDR image from the captured images and relative exposure
relExposure = exposure./exposure(1); %relative exposure
HDRimg= makehdr(images,'RelativeExposure',relExposure);
rgb = tonemap(HDRimg); %convert to color image
imshow(rgb) %display


% Simulate a low resolution noisy image from the HDR image
blurredImage = imgaussfilt(rgb,3);  % Apply Gaussian blur
blurredImageP = imnoise(blurredImage,'salt & pepper'); %Apply salt and pepper noise
lowResImageAdvanced = imresize(blurredImageP, 0.2, 'bicubic');
%lowResImagePristine = imresize(rgb, 0.2, 'bicubic'); %Create an unnoisy
%low resolution image as reference for PSNR and SSIM calculations later
% Display the original image vs. resulting noisy blurred downsized image
figure;
subplot(1,2,1);
imshow(rgb);
title('Original Image');
subplot(1,2,2);
imshow(lowResImageAdvanced);
title('Simulated Low-Resolution Image');
%Results saved in "HDR_vs_simulated_low_res.jpg" in "Project 2 - Demo"




% % Load a pre-trained DnCNN network to denoise the picture
net = denoisingNetwork('DnCNN');
% Split a noisy image into three channels and denoise each channel
[noisyR,noisyG,noisyB] = imsplit(lowResImageAdvanced);
denoisedR = denoiseImage(noisyR,net);
denoisedG = denoiseImage(noisyG,net);
denoisedB = denoiseImage(noisyB,net);
%combine the denoised channels back into a colored image
denoisedImg = cat(3,denoisedR,denoisedG,denoisedB);
%Display the original and denoised images
figure;
subplot(1,2,1);
imshow(lowResImageAdvanced);
title('Noisy Image');
subplot(1,2,2);
imshow(denoisedImg);
title('Denoised Image');
%Result is shown in "noisy_vs_dncnnDenoised.jpg" in "Project 2 - Demo"


% %Option to compare and contrast the PSNR and SSIM of noisy and denoised
% %images
% noisyPSNR = psnr(lowResImageAdvanced,lowResImagePristine);
% fprintf("\n The PSNR value of the noisy image is %0.4f.",noisyPSNR);
% 
% denoisedPSNR = psnr(denoisedImg,lowResImagePristine);
% fprintf("\n The PSNR value of the denoised image is %0.4f.",denoisedPSNR);
% 
% noisySSIM = ssim(lowResImageAdvanced,lowResImagePristine);
% fprintf("\n The SSIM value of the noisy image is %0.4f.",noisySSIM);
% 
% denoisedSSIM = ssim(denoisedImg,lowResImagePristine);
% fprintf("\n The SSIM value of the denoised image is %0.4f.",denoisedSSIM);


% Use ESRGAN to double the resolution of the low resolution image
imgSR = ESRGAN_2xSuperResolution(lowResImageAdvanced);
imwrite(imgSR, "Bear_2x_ESRGAN.png"); 
% Display the original and upscaled images
figure;
subplot(1,2,1);
imshow(lowResImageAdvanced);
title('No denoising Low-Resolution Image');
subplot(1,2,2);
imshow(imgSR);
title('ESRGAN Upscaled Image');
%Result is shown in "noisy_vs_dncnnDenoised.jpg" in "Project 2 - Demo"

%The part on VGG-19 network to adapt the style of content image to an input
% style image has been omitted as the code is from the sources:
% https://www.mathworks.com/help/images/neural-style-transfer-using-deep-learning.html