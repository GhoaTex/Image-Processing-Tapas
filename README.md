# Image-Processing-Tapas
A collection of mini projects on image processing using a variety of computational tools and readily available office items.


1- Real Time Video Editing in Python

  _Goal:_ Create a script leveraging the powerful Python Numpy and OpenCV libraries to develop a real-time video enhancement routine with edge detection. The raw and edited videos are displayed side by side and the user has the option of saving the video in MP4 files using the MPEG-4 encoding standard through an established GStreamer video pipeline in the current file directory. Auto exposure control, white balance correction and color enhancement are used to enhance image quality. Edge detection is implemented using contouring, edge enhancement and morphological operations. 

  _Code:_** Project 1 - Real Time Video Editing.py**
  
  _Results:_ **Project 1 - Demo** file contains frames from compare and contrasting the edited and raw versions of images being captured by a Logitech HD 720p webcam. The processed images clearly were enhanced compared to the raw images even in poor lighting conditions. Edge detection works well on objects with smooth surfaces and high color contrast e.g. Car_comparison.png and Stationary_comparison.png Textured surfaces (e.g. Bear_comparison.png) and surfaces with color variation (Floral_mirror.png) were more challenging for the edge-detection based contouring to be accurate .
