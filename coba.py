import cv2
import time
import math

# Open video file
cap = cv2.VideoCapture("istockphoto-1184900033-640_adpp_is.mp4")

# Initialize background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Counter for cars
car_count = 0

# Threshold for minimum distance between object centroids to consider it the same object
MIN_DIST_THRESHOLD = 50

# Delay and frame display frequency settings
frame_delay = 0.1  # Delay between frames in seconds (100 ms)
display_every_n_frames = 2  # Display every 2nd frame

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % display_every_n_frames != 0:
        continue

    # Apply background subtraction
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours of moving objects
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List to store detected car centroids
    car_centroids = []

    # Reset car count for this frame
    frame_car_count = 0

    # Analyze each contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Assume vehicle based on aspect ratio (adjust threshold as needed)
            if aspect_ratio > 3.5:  # Adjust aspect ratio threshold as needed
                # Calculate centroid of the bounding box
                cx = x + w // 2
                cy = y + h // 2

                # Check if this centroid is close to any existing car centroid
                duplicate_found = False
                for centroid in car_centroids:
                    dist = math.sqrt((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2)
                    if dist < MIN_DIST_THRESHOLD:
                        duplicate_found = True
                        break

                # If no duplicate found, consider it as a new car
                if not duplicate_found:
                    frame_car_count += 1
                    car_count += 1
                    car_centroids.append((cx, cy))

                    # Draw bounding box for visualization
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display car count on frame
    cv2.putText(frame, f'Cars: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display results
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Delay to slow down processing
    time.sleep(frame_delay)

    # Exit if ESC key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Total Cars:", car_count)
