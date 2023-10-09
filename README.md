# Video Processing with OpenCV

This Python script utilizes OpenCV to process video frames, applying various computer vision techniques to create a modified video with added effects. The code performs face detection, frame rotation, blurring, watermarking, and overlaying frames from a separate video.

## Prerequisites

- Python 3
- OpenCV
- NumPy

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

2. Install Dependencies:
   ```bash
   pip install opencv-python numpy

4. Run the Script:
   ```bash
   python video_processing.py

6. Input Videos:

7. Place the input videos (exercise.mp4 and talking.mp4) in the same directory as the script.
Output Video:

The processed video will be saved as FinalResult_exercise.avi in the same directory.

# Parameters and Customization
Modify the parameters in the script, such as face detection parameters (para1 and para2), input video names, watermark filenames, etc., based on your requirements.

# Explanation of Code
Face Detection: Haar Cascade face detection is used with different parameters for different stages of the video.

Frame Rotation and Blurring: Frames are rotated by specified angles (0, -25, 25), and faces in the rotated frames are blurred using GaussianBlur.

Watermarking: Two watermarks are added to frames based on certain conditions related to the frame count.

Talking Frame Overlay: Frames from the "talking" video are overlaid on the main video, adding a border and resizing them.

Output Writing: Processed frames are written to the output video file.
