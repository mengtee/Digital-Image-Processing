"""
@Students: Chin Yu Wei, Lai Yi Mei, Tee Meng Kiat, Than Hui Ru
"""
import numpy as np
import cv2

# Read and write the video file
# To processed the video, change the input size of the original video
# Original video stand for input video (street, exercise and office )
ori_vid = cv2.VideoCapture("exercise.mp4")
talk_vid = cv2.VideoCapture("talking.mp4")
final_output = cv2.VideoWriter('FinalResult_exercise.avi',cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1920,1080))
# Change the parameter base on the report
para1 = {'scaleFactor': 1.1, 'minNeighbors': 12, 'maxSize': (100, 100)}
para2 = {'scaleFactor': 1.1, 'minNeighbors': 12, 'maxSize': (280, 280)}

# original video size, get the layout(height, width and total frame) of the video
input_vid_height = int(ori_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_vid_width = int(ori_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
ori_total_no_frames = int(ori_vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Import watermark
watermark1 = cv2.imread('watermark1.png')
watermark2 = cv2.imread('watermark2.png')

# Resize the watermark to match the input video size
resized_watermark1 = cv2.resize(watermark1, (input_vid_width,input_vid_height),interpolation=cv2.INTER_AREA)
resized_watermark2 = cv2.resize(watermark2, (input_vid_width,input_vid_height),interpolation=cv2.INTER_AREA)

# User defined function for the determing the tilted faces
# This function is use to rotate the input frame, consist of two parameter, frame and angle
def rotate_frame(frame, angle):
    if angle == 0: 
        return frame
    height, width = frame.shape[:2]
    # matrix to rotate the image, with the parameter of (center of rotation,
    # angle of rotation and scaling factor)
    r_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.5)
    
    # Applying afine transformation for the image with the rotational matrix
    result = cv2.warpAffine(frame, r_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return result

# This function is use to detect the new position (x,y,w,h) of the rotated faces 
def rotate_point(point, frame, angle):
    if angle == 0: 
        return point
    # Operation to determine the new rotation value
    x = point[0] - frame.shape[1]*0.4
    y = point[1] - frame.shape[0]*0.4
    newx = x*np.cos(np.radians(angle)) + y*np.sin(np.radians(angle)) + frame.shape[1]*0.4
    newy = -x*np.sin(np.radians(angle)) + y*np.cos(np.radians(angle)) + frame.shape[0]*0.4
    return int(newx), int(newy), point[2], point[3]

# Loop through all the frame in the based video
for frame_count in range(0, ori_total_no_frames):
    # Read each of the frame, return 2 value, Boolean and the frame
    success_vid, frame_vid= ori_vid.read()
    
    # Face detection using face cascade model, reading the Haar Cascade model
    face_cascade = cv2.CascadeClassifier("face_detector.xml")
    # For loop to process different angle
    for angle in [0, -25, 25]:
        # Rotate the frame using the user-defined function
        rframe = rotate_frame(frame_vid, angle)
        # detectMultiScale take in 4 parameter, which are the frame to identify faces,
        # Scaling factor, minNeighbour
        # Return a list of retangle(x,y,w,h) in which faces is detected
        #If statement to control the max face size from the first and second half of the video
        if (frame_count <= int(ori_total_no_frames/2)):
            faces = face_cascade.detectMultiScale(rframe, **para1)
        
        if (frame_count >= int(ori_total_no_frames/2)):
            faces = face_cascade.detectMultiScale(rframe, **para2)
        # If the rotated faces is detected
        # detected will return 4 value, x,y,w,h
        if angle == 0:
            for (x, y, w, h) in faces:
                face = frame_vid[y:y+h, x:x+w]
                blur_face = cv2.GaussianBlur(face,(99,99),0)
                frame_vid[y:y+h, x:x+w] = blur_face
        else:
            # Use to determine the new face position(tilted faces) and blur it.
            for (x, y, w, h) in faces:
                pos = (x,y,w,h)
                # Use the function in determine the new position of the original image
                faces = [rotate_point(pos, frame_vid, -angle)]
                for (x,y,w,h) in faces:
                    # Blur the face based on the new position
                    face = frame_vid[y:y+h, x:x+w]
                    blur_face = cv2.GaussianBlur(face,(99,99),0)
                    frame_vid[y:y+h, x:x+w] = blur_face   
    
    # Get the frame count and read the talking video 
    talk_total_no_frames = talk_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    success_talk, frame_talk = talk_vid.read()
    
    #If there is a talking frame, run the operation
    if success_talk == True:
        # Adding border to the frame
        border_frame = cv2.copyMakeBorder(frame_talk,25,25,25,25,cv2.BORDER_CONSTANT)
        # Resize all of the frame
        resized_frame = cv2.resize(border_frame, (int(input_vid_width/4), int(input_vid_height/4)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        dimention = resized_frame.shape
        frame_vid[0:int(dimention[0]),0:int(dimention[1])] = resized_frame
    
    # Adding watermark to different frame
    if (success_vid == True) and (frame_count <= int(ori_total_no_frames/2)):
        final = cv2.addWeighted(frame_vid, 1, resized_watermark1,1, 0)
        frame_vid = final
        
    if (success_vid == True) and (frame_count > int(ori_total_no_frames/2)):
        final = cv2.addWeighted(frame_vid, 1, resized_watermark2,1, 0)
        frame_vid = final
    
    # Write the output, save the video
    final_output.write(frame_vid)