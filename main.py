import cv2
import numpy as np

def detect_object(frame, object_image, threshold=0.8):
    """
    Detects an object in a frame based on an image of it.
    Returns the top-left and bottom-right coordinates of the object.
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate( frame,object_image, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # If the object is found
    if max_val > threshold:
        # Get the coordinates of the object
        top_left = max_loc
        bottom_right = (top_left[0] + object_image.shape[1], top_left[1] + object_image.shape[0])
        return top_left, bottom_right
    else:
        return None, None

def draw_rectangle(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2):
    """
    Draws a rectangle on the frame.
    """
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

def process_video_stream(object_image_path, threshold=0.8, wait_time=1):
    """
    Processes a video stream and detects an object based on an image of it.
    """
    # Load the image of the object
    object_image = cv2.imread(object_image_path)

    # Open a video stream
    cap = cv2.VideoCapture(0)
    #frame = cv2.imread("Untitled.jpg")

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Detect the object in the frame
        top_left, bottom_right = detect_object(frame, object_image, threshold)

        # If the object is found, draw a rectangle around it
        if top_left is not None:
            draw_rectangle(frame, top_left, bottom_right)

        # Display the frame
        cv2.imshow('Detected Object', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # Release the video stream
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video_stream('shape1.jpg')