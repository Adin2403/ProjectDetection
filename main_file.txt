import cv2
import numpy as np

def detect_object(frame, shape_img, sift):
    """
    Detects an object in the given frame based on the given reference image and SIFT detector.
    Returns the dimensions of the object in pixels, or (0, 0) if the object is not detected.
    """
    # Extract keypoints and descriptors from the frame
    kp_frame, des_frame = sift.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Use the Brute-Force Matcher to find matches between the frame and the reference image
    #bf = cv2.BFMatcher()

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch( des_frame,des_shape,  k=2)
   # matches = bf.knnMatch( des_frame,des_shape, k=2)

    # Filter out poor matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance: #0.7
            good_matches.append(m)

    # Calculate the dimensions of the object in pixels
    width_pixels = 0
    height_pixels = 0
    if len(good_matches) > 4:
        src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_shape[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = shape_img.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            width_pixels = int(abs(dst[0][0][0] - dst[3][0][0]))
            height_pixels = int(abs(dst[0][0][1] - dst[1][0][1]))

    return (width_pixels, height_pixels, frame)

def calculate_dimensions(width_pixels, height_pixels, focal_length, distance):
    """
    Calculates the dimensions of the object in centimeters based on the given dimensions in pixels,
    focal length of the camera in pixels, and distance of the object from the camera in meters.
    """
    width_cm = (width_pixels / focal_length) * distance * 100
    height_cm = (height_pixels / focal_length) * distance * 100
    return (width_cm, height_cm)

if __name__ == "__main__":

    # Load the reference image with the object shape
    shape_img = cv2.imread('shape.jpg', 0)

    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    # Extract keypoints and descriptors from the reference image
    kp_shape, des_shape = sift.detectAndCompute(shape_img, None)

    # Set up the video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the object in the frame
        width_pixels, height_pixels, frame = detect_object(gray, shape_img, sift)

        # Calculate the dimensions of the object in centimeters
        focal_length = 1200.0  # Focal length of the camera in pixels
        distance = 0.5  # Distance of the object from the camera in meters
        width_cm, height_cm = calculate_dimensions(width_pixels, height_pixels, focal_length, distance)

        # Draw the dimensions on the frame
        cv2.putText(frame, f'Width: {width_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f'Height: {height_cm:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()
