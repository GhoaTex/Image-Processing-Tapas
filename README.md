# Image-Processing-Tapas
A collection of mini projects on image processing using a variety of computational tools and readily available office items.


Project 1- Real Time Video Editing in Python

  _Goal:_ Create a script leveraging the powerful Python Numpy and OpenCV libraries to develop a real-time video enhancement routine with edge detection. The raw and edited videos are displayed side by side and the user has the option of saving the video in MP4 files using the MPEG-4 encoding standard through an established GStreamer video pipeline in the current file directory. Auto exposure control, white balance correction and color enhancement are used to enhance image quality. Edge detection is implemented using contouring, edge enhancement and morphological operations. 

  _Code:_ **Project 1 - Real Time Video Editing.py**
  
  _Results:_ **Project 1 - demo** file contains frames from compare and contrasting the edited and raw versions of images being captured by a Logitech HD 720p webcam. The processed images clearly were enhanced compared to the raw images even in poor lighting conditions. Edge detection works well on objects with smooth surfaces and high color contrast e.g. Car_comparison.png and Stationary_comparison.png Textured surfaces (e.g. Bear_comparison.png) and surfaces with color variation (Floral_mirror.png) were more challenging for the edge-detection based contouring to be accurate .

Project 2 - Sensor Analysis and Deep Learning(DL) Assisted Image Processing in MATLAB

  _Goal:_ Create a MATLAB script using Image Acquisition Toolbox, Image Processing Toolbox, Computer Vision Toolbox, Deep learning Toolbox, ESRGAN Single Image Super
Resolution Matlab Port to report on the camera's dynamic range and signal-to-noise ratio as well as composing a HDR picture from shots of the same scene under different lighting conditions. The HDR image is then treated with Gaussian blurred, salt and pepper noise to and lowered in resolution scaled to 20% of its original size using bicubic downsampling method. The noisy lowered resolution picture is then noise filtered using pre-trained DnCNN network, enhanced in resolution using the pre-trained ESRGAN X2 and adapted to the styles of famous painters Renoir and Dali using the pre-trained VGG-19 network. Optional peak signal-to-noise ratio (PSNR) and structural similarity index measure(SSIM) can be calculated for the denoised picture compared to the original

  _Code:_ **Project 2 - Sensor Analysis and DL Assisted Image Processing.m**
  
  _Results:_  The Logitech HD 720p webcam yields a consistent dynamic range of 255 and a SNR of 3.15 dB averaged over three trials. **Project 2 - demo** file contains frames from compare and contrasting the edited and raw versions of images being captured by a Logitech HD 720p webcam. The processed images clearly were enhanced compared to the raw images even in poor lighting conditions. Edge detection works well on objects with smooth surfaces and high color contrast e.g. Car_comparison.png and Stationary_comparison.png Textured surfaces (e.g. Bear_comparison.png) and surfaces with color variation (Floral_mirror.png) were more challenging for the edge-detection based contouring to be accurate .
