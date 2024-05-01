
import numpy as np
import cv2
"""This script uses Python's powerful numpy and OpenCV libraries to develop a real-time video enhancement routine with edge detection. 
The raw and edited videos are displayed side by side and the user has the option of saving the video in MP4 files using the MPEG-4 encoding
standard through an established GStreamer video pipeline in the current file directory."""


#Can check openCV build information
#info = cv2.getBuildInformation()
#print(info)
#Define color correction and enhancement image processing functions to be used on individual frames

def apply_clahe(frame):
    """Auto exposure control using contrast limited adaptive histogram equalization (CLAHE). This also helps
    reduce over-amplification of noise in homogeneous regions of the image."""

    #Convert to LAB color space and split channels
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    #Apply CLAHE to luminance L channel, enhancing local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #Merge channels and convert back to BFR color space
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final



def apply_gray_world(frame):
    """ White balance correction using the Gray World assumption that the average of the resulting colors would be
    # neutral gray so each color channel would yield the same average as each of the two other channels."""

    # For a given frame, convert to float32 and calculate the average of each of the BGR color channels.
    float_frame = np.float32(frame)
    avg_bgr = np.mean(float_frame, axis=(0, 1))

    # Average of individual mean of the color channels and divide by the mean of each channel respectively to get the
    # scaling factor corresponding to each channel and then scale the values of each channel accordingly
    avg_all = np.mean(avg_bgr)
    scale_factors = avg_all / avg_bgr

    float_frame[:, :, 0] *= scale_factors[0]  # Blue channel
    float_frame[:, :, 1] *= scale_factors[1]  # Green channel
    float_frame[:, :, 2] *= scale_factors[2]  # Red channel

    # Recast the values to the valid range [0, 255] and convert back to uint8
    GWcorrected_frame = np.clip(float_frame, 0, 255).astype(np.uint8)

    return GWcorrected_frame


def enhance_color(frame, saturation_scale=1.3, value_scale=1.1):
    """Enhance the BGR colors of an input frame by adjusting its saturation and value using saturation_scale and
    value_scale. >1 to increase, <1 to decrease.    """

    # Convert the image from BGR to HSV color space and convert to float32 type, split the HSV channels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = np.float32(hsv)
    h, s, v = cv2.split(hsv)

    # Scale the S and V channels
    s *= saturation_scale
    v *= value_scale

    # Recast values to the valid range [0, 255]
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)

    # Merge the channels back, convert to uint* and then back to BFR color space
    hsv_enhanced_frame = cv2.merge([h, s, v])
    hsv_enhanced_frame = np.uint8(hsv_enhanced_frame)
    bgr_enhanced_frame = cv2.cvtColor(hsv_enhanced_frame, cv2.COLOR_HSV2BGR)

    return bgr_enhanced_frame

##[Uncomment to implement saving video functionality]
# # Define and initialize GStreamer pipeline for video capture
# pipeline = ('autovideosrc !''video/x-raw, width=(int)640, height=(int)480, framerate=(fraction)30/1 !'
#     'videoconvert !''appsink drop=true sync=false')
# cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

#Initiate video capture
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

##[Uncomment to implement saving video functionality]
# # Get the width and height of the captured video frames
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) else 30  # Use 30 FPS as default if FPS is 0
# # Define codec for MPEG-4
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is for MPEG-4
# # Set up the video writers for both unedited and edited versions of video
# out_raw = cv2.VideoWriter('no_edit.mp4', fourcc, fps, (frame_width, frame_height))
# if not out_raw.isOpened():
#     print("Error: Could not open video writer for no_edit.mp4")
# out_edited = cv2.VideoWriter('edited.mp4', fourcc, fps, (frame_width, frame_height))
# if not out_edited.isOpened():
#     print("Error: Could not open video writer for edited.mp4")


#loop to process Real-Time video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # #wirte unedited frame into raw video
    # out_raw.write(frame)

    # Apply processing algorithms each in turn
    autoExp_frame = apply_clahe(frame)
    autoExp_GW_frame=apply_gray_world(autoExp_frame)
    processed_frame=enhance_color(autoExp_GW_frame)

    # Contour detection in real-time:
    # Convert to grayscale for contour detection and blur the image to reduce noise
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame= cv2.bilateralFilter(gray_frame, 5, 60, 60)

    # Enhance edges using the Sobel operator to detect high intensity change areas, then calculate the gradient and
    # combined with a binary threshold to highlight edge areas
    sobelx = cv2.Sobel(blurred_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_frame, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobelx, sobely)
    _, binary = cv2.threshold(np.uint8(gradient), 40, 255, cv2.THRESH_BINARY)

    # Use a kernel to perform morphological operations: "closing" is the resulting image from holes being filled
    # and "opening" is for when noise gets reduced for greater object separation used to find contourx
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Find contours and use edge-preserving filtering to enhance details of the image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enhanced_frame = cv2.detailEnhance(processed_frame, sigma_s=15, sigma_r=1)

    # Draw contours on the processed frame
    cv2.drawContours(enhanced_frame, contours, -1, (0, 255, 0), 2)  # -1 means draw all contours

    # # Write the processed frame to the processed video file
    # out_edited.write(enhanced_frame)

    #Display both processed and unprocessed images
    cv2.imshow('Raw Frame', frame)
    cv2.imshow('Processed Frame with contours',enhanced_frame)

    if cv2.waitKey(2) & 0xFF == ord('x'):
        break

cap.release()


cv2.destroyAllWindows()
