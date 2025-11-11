# Import OpenCV for all computer vision functions such as image capture, detection, and display
# We use OpenCV because it provides built-in tools for real-time image analysis and camera handling
import cv2

# Import deque from Python's standard library
# We use deque to keep a short, fixed-length history of results (like recent smile detections)
# This allows us to smooth out noisy frame-by-frame predictions
from collections import deque

# Import CSV module for logging emotion data to a file
import csv

# Import datetime for timestamping emotion logs
from datetime import datetime

# Import os for checking if log file exists
import os


# Define a helper function to draw text on an image frame
# We create this function so we can easily display labels or instructions without repeating code
def put_text(img, text, org=(30, 40), scale=1.0, color=(0, 255, 0), thick=2):
    # cv2.putText draws text directly on an image
    # img: the frame we’re drawing on
    # text: the text string to display
    # org: (x, y) coordinates for the bottom-left corner of the text
    # We use (30, 40) as a default so the text appears clearly near the top-left corner
    # scale: controls the font size
    # color: color of the text in BGR format (Blue, Green, Red)
    # thick: controls the stroke width of the text
    # cv2.LINE_AA: enables anti-aliasing for smoother text edges
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# Function to initialize emotion log CSV file
def init_emotion_log(log_file="emotion_log.csv"):
    # Check if the log file already exists
    # If it doesn't exist, create it with headers
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Expression', 'Happiness Score (%)'])
        print(f"Created new emotion log file: {log_file}")
    else:
        print(f"Using existing emotion log file: {log_file}")


# Function to log emotion data to CSV file
def log_emotion(log_file, expression, score):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert score to percentage
    score_percent = round(score * 100, 2)
    
    # Append data to CSV file
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, expression, score_percent])


# Main function that handles face and smile detection
def run_face():
    # Initialize emotion log file
    log_file = "emotion_log.csv"
    init_emotion_log(log_file)
    
    # Load OpenCV's built-in Haar Cascade XML files for detecting faces and smiles
    # Haar cascades are pre-trained models that detect specific visual patterns like faces or smiles
    # Using them saves time because we don't need to train our own models
    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    smile_xml = cv2.data.haarcascades + "haarcascade_smile.xml"

    # Create classifier objects from the XML files
    # These classifiers are later used to run the actual detection on the video frames
    face_cascade = cv2.CascadeClassifier(face_xml)
    smile_cascade = cv2.CascadeClassifier(smile_xml)

    # Start video capture from the default webcam (index 0)
    # The VideoCapture object allows OpenCV to access the camera feed
    cap = cv2.VideoCapture(0)

    # Check that the camera opened correctly
    # This prevents the program from crashing if the camera is not available or permission is denied
    if not cap.isOpened():
        print("Could not open camera. Allow camera access in System Settings → Privacy → Camera.")
        return

    # Create a deque to store recent smile detections
    # A fixed window of 15 frames helps smooth the results, so one missed detection doesn't cause flickering
    happy_votes = deque(maxlen=15)
    
    # Counter for logging interval (log every N frames to avoid too many entries)
    frame_count = 0
    log_interval = 30  # Log every 30 frames (approximately once per second at 30fps)

    # Start the main loop for real-time frame processing
    while True:
        # Read one frame from the webcam
        ok, frame = cap.read()

        # If reading fails (camera disconnected, etc.), stop the loop
        if not ok:
            break

        # Flip the frame horizontally so it behaves like a mirror
        # This makes the video more natural for users viewing themselves
        frame = cv2.flip(frame, 1)

        # Convert the frame from color (BGR) to grayscale
        # Haar cascades operate faster and more accurately on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces within the grayscale frame
        # detectMultiScale scans the image for objects that match the trained face pattern
        # scaleFactor: how much the image size is reduced at each scale
        # minNeighbors: how many overlapping detections are needed to confirm a face
        # minSize: ignores detections smaller than a given size
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
        )

        # Default label and color used if no face is detected
        label = "no face"
        color = (0, 0, 255)  # red text for “no face”

        # Process only the first detected face for simplicity
        for (x, y, w, h) in faces[:1]:
            # Draw a rectangle around the detected face
            # This gives a visual cue that detection is working
            #parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 255), 1)

            # Extract the region of interest (ROI) that contains only the face
            # This makes smile detection faster and more accurate because it only runs on the face area
            #parameters: image, y-start:y-end, x-start:x-end
            #this is where we are cropping the face from the rest of the image
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Focus smile detection on the lower half of the face
            # This reduces false positives caused by shadows or eye wrinkles
            #parameters: y-start:y-end, x-start:x-end
            mouth_region = roi_gray[h // 2 : h, 0:w]

            # Detect smiles inside the mouth region
            # The parameters are tuned for stable detection:
            # - A higher scaleFactor (1.4) reduces sensitivity to small changes
            # - A higher minNeighbors (18) reduces false positives
            # - A minimum size prevents very small random detections
            smiles = smile_cascade.detectMultiScale(
                mouth_region,
                scaleFactor=1.4,
                minNeighbors=18,
                minSize=(30, 30)
            )

            # If at least one smile is found, mark the frame as “happy”
            is_happy = len(smiles) > 0

            # Store 1 for happy, 0 for not happy, in our deque history
            # This keeps a short memory of previous frames for smoothing
            happy_votes.append(1 if is_happy else 0)

            # Calculate an average happiness score from recent frames
            # Using an average prevents the label from flickering if a smile disappears briefly
            score = sum(happy_votes) / max(1, len(happy_votes))

            # Classify expression based on score
            # If more than half the recent frames show a smile, we label as happy
            label = "happy" if score >= 0.5 else "sad"

            # Use color to make the result more readable:
            # green for happy, yellow for sad
            color = (0, 255, 0) if label == "happy" else (0, 255, 255)

            # Only process the first face to keep the program simple and consistent
            break

        # Log emotion data periodically (not every frame to keep file size manageable)
        frame_count += 1
        if frame_count >= log_interval and label != "no face":
            log_emotion(log_file, label, score)
            frame_count = 0  # Reset counter

        # Display the detected expression on the video feed
        put_text(frame, f"Expression: {label}", (30, 40), 1.0, color, 2)
        
        # Show happiness score percentage if a face is detected
        if label != "no face":
            score_text = f"Happiness: {int(score * 100)}%"
            put_text(frame, score_text, (30, 80), 0.7, (255, 255, 255), 2)

        # Show an instruction line near the bottom of the frame
        # We calculate y position dynamically so it always stays near the bottom regardless of resolution
        put_text(frame, "Press q to quit", (30, frame.shape[0] - 20), 0.6, (180, 180, 180), 1)

        # Display the frame in a window
        # “imshow” refreshes continuously to show real-time detection results
        cv2.imshow("Face: Happy vs Sad", frame)

        # Wait for 1 millisecond for a key press
        # If the 'q' key is pressed, exit the loop
        # We use & 0xFF to ensure compatibility across platforms
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When finished, release the webcam to free system resources
    cap.release()

    # Close all OpenCV display windows cleanly
    cv2.destroyAllWindows()
    
    # Notify user where the emotion log is saved
    print(f"\nEmotion data has been logged to: {log_file}")
    print("You can open this file in Excel or any spreadsheet application to view your emotion history.")


# Only run the detection if this file is executed directly (not imported)
# This prevents accidental execution if someone imports this file into another script
if __name__ == "__main__":
    run_face()
