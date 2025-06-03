from ultralytics import YOLO
import cv2
import os

# Load your trained model (updated path for your workspace)
model_path = "best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    exit(1)

model = YOLO(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Optional: set resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

print("Starting YOLO detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Run YOLOv8 prediction
    results = model(frame, conf=0.5)

    # Annotate results directly on the frame
    annotated_frame = results[0].plot()  # Automatically draws boxes and labels

    # Show the frame
    cv2.imshow("YOLOv8 Phone Detection", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.") 