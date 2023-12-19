import numpy as np
import cv2
import datetime
import csv
import math
from collections import deque

def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices

# Important!
# Some lines of code need to be updated so the output data is saved where you want it to be
# therefore, check and adjust line 29, 69, and 241 before you run the code


# load video file
cap = cv2.VideoCapture(0)

# Generate a timestamp string
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Specify the output filename with the timestamp
# In the following format (might need to be changed depending on OS of computer):
# f"/Users/yourname/nameoffolder/videoname_{timestamp}.mp4"
output_filename = f"/Users/Ingvild/screentracker/output_video_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get the webcam stream dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the VideoWriter object to save the video
video_out = cv2.VideoWriter(output_filename, fourcc, 10, (frame_width, frame_height))

# Create a list to store the alignment data
alignment_data = []

# Variables to keep track of the previous second and alignment status
previous_alignment_status = None
previous_second = None

# min_marker_perimeter = 0.03
# max_marker_perimeter = 0.05

# setting up dictionary of aruco markers and adding parameters for detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
# parameters.minMarkerPerimeterRate = min_marker_perimeter
# parameters.minMarkerPerimeterRate = max_marker_perimeter

# refines the parameters for better detection
parameters.cornerRefinementMethod = 3
parameters.errorCorrectionRate = 0.5

# Preprocessing parameters
resize_width = 640  # Adjust the desired width
blur_kernel_size = (7, 7)  # Adjust the kernel size for blurring
threshold_value = 150  # Adjust the threshold value
morph_kernel_size = (7, 7)  # Adjust the kernel size for morphological operations

# load reference image
# remember to use reference image of actual tablet, it makes a difference!
# upload reference image to a folder and specify the filepath like this:
# "/Users/yourname/nameoffolder (if image is in a subfolder)/imagename.jpeg" (or .jpg, please check!)
refImage = cv2.cvtColor(cv2.imread("/Users/Ingvild/ipad6.jpeg"), cv2.COLOR_BGR2GRAY)

# detect markers in reference image
refCorners, refIds, refRejected = cv2.aruco.detectMarkers(refImage, aruco_dict, parameters=parameters)
# create bounding box from reference image dimensions
rect = np.array([[[0, 0],
                  [refImage.shape[1], 0],
                  [refImage.shape[1], refImage.shape[0]],
                  [0, refImage.shape[0]]]], dtype="float32")

# noise reduction
h_array = deque(maxlen=5)

while True:
    # read next frame from VideoCapture
    ret, frame = cap.read()
    if frame is not None:
        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, blur_kernel_size, 0)

        # Apply thresholding to convert the image to binary format
        _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to refine the marker regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # detect aruco markers in gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # if markers were detected
        if res_ids is not None and len(res_ids) > 0:
            # find which markers in frame match those in reference image
            idx = which(refIds, res_ids)
            alignment_status = "Tablet aligned"

            # Make an empty list for the rotation angle and corner angles
            rotation_angle = 0.0
            angles = [0.0, 0.0, 0.0, 0.0]

            # if any detected marker in frame is also in the reference image
            if len(idx) > 0:
                # flatten the array of corners in the frame and reference image
                these_res_corners = np.concatenate(res_corners, axis=1)
                these_ref_corners = np.concatenate([refCorners[x] for x in idx], axis=1)

                # Estimate homography matrix
                if these_res_corners.shape[0] == these_ref_corners.shape[0]:
                    try:
                        h, s = cv2.findHomography(these_ref_corners, these_res_corners, cv2.RANSAC, 5.0)

                    # in case the program cannot calculate the homography matrix
                    except cv2.error as e:
                        print("Homography calculation failed:", str(e))
                        alignment_status = "Homography calculation failed"

                    else:
                        # for smoothing
                        h_array.append(h)
                        this_h = np.mean(h_array, axis=0)

                        # transform the rectangle using the homography matrix
                        newRect = cv2.perspectiveTransform(rect, this_h, (gray.shape[1], gray.shape[0]))

                        # draw the rectangle on the frame
                        frame = cv2.polylines(frame, np.int32(newRect), True, (0, 0, 0), 10)

                        # Calculate the rotation angle
                        top_vector = newRect[0][1] - newRect[0][0]
                        rotation_angle = math.degrees(math.atan2(top_vector[1], top_vector[0]))

                        # Display the rotation angle
                        cv2.putText(frame, f"Rotation Angle: {rotation_angle:.2f} degrees", (50, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # calculate the angles at each corner of the quadrilateral
                        rect_points = np.squeeze(newRect)

                        # calculate the angles between the lines
                        angles = []
                        for i in range(4):
                            pt1 = rect_points[i]
                            pt2 = rect_points[(i + 1) % 4]
                            pt3 = rect_points[(i + 2) % 4]

                            # calculate the vectors of the lines
                            v1 = pt1 - pt2
                            v2 = pt3 - pt2

                            # calculate the angle between the lines using dot product
                            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                            angle_deg = np.degrees(angle)
                            angles.append(angle_deg)

                        # check if any angle is above 100 degrees and if rotation angle is between -30 and 30 degrees
                        # These values need to be updated depending on what values are optimal
                        if any(angle > 100 for angle in angles) or rotation_angle > 30 or rotation_angle < -30:
                            cv2.putText(frame, "Tablet not aligned", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            alignment_status = "Tablet not aligned"

                        # if angles are not above 100 deg and rotation angle is okay, tablet is aligned
                        else:
                            cv2.putText(frame, "Tablet aligned", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            alignment_status = "Tablet aligned"

                    # display the angles on the screen
                    for i, angle in enumerate(angles):
                        angle_text = f"Angle {i + 1}: {angle:.2f} degrees"
                        cv2.putText(frame, angle_text, (50, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    # No ArUco markers found
                    # Display text on screen
                    cv2.putText(frame, "No ArUco markers detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

                    # No ArUco markers detected
                    # Update alignment status
                    alignment_status = "No ArUco markers detected"
                    continue

            # Updating alignment status once every second
            if alignment_status != previous_alignment_status:
                current_time = datetime.datetime.now()
                current_second = current_time.second

                # Making sure information is only appended to the list once every second
                if current_second != previous_second or previous_second is None:
                    alignment_data.append(
                        [current_time.strftime("%Y-%m-%d %H:%M:%S"), alignment_status, rotation_angle] + angles)
                    previous_second = current_second

        else:
            # No ArUco markers found
            cv2.putText(frame, "No ArUco markers detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            # Update alignment atatus to No ArUco markers detected
            alignment_status = "No ArUco markers detected"

            # Get the current timestamp
            current_time = datetime.datetime.now()
            current_second = current_time.second

            # Check if the current second is different from the previous second
            if current_second != previous_second or previous_second is None:
                # Append the alignment data for the current second
                alignment_data.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), alignment_status])

                # Update the previous second and alignment status
                previous_second = current_second
                previous_alignment_status = alignment_status

        # draw detected markers in frame with their ids
        cv2.aruco.drawDetectedMarkers(frame, res_corners, res_ids)

        # Write the processed frame to the output video
        video_out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # exit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Save the data to a CSV file
# again, update filepath to where you want to save the csv data
tracker_data_csv_filename = f"/Users/Ingvild/screentracker/tracker_data_{timestamp}.csv"
with open(tracker_data_csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Alignment Status", "Rotation Angle", "Upper Left Angle", "Upper Right Angle",
                     "Lower Right Angle", "Lower Left Angle"])
    writer.writerows(alignment_data)

# When everything is done, release the capture and video output
if video_out is not None:
    video_out.release()
cap.release()

# close cv2 window
cv2.destroyAllWindows()
