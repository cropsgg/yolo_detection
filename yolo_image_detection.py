from ultralytics import YOLO
import cv2
import os

# Load your trained model
model_path = "best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    exit(1)

print("Loading YOLO model...")
model = YOLO(model_path)
print("Model loaded successfully!")

# Test with a test image or create a simple test
print("Testing model with camera image...")

# Try to capture one frame to test the model
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Camera access successful! Running detection...")
        
        # Run YOLOv8 prediction
        results = model(frame, conf=0.5)
        
        # Annotate results directly on the frame
        annotated_frame = results[0].plot()
        
        # Save the result
        cv2.imwrite("detection_result.jpg", annotated_frame)
        print("Detection result saved as 'detection_result.jpg'")
        
        # Print detection results
        for r in results:
            print(f"Detected {len(r.boxes)} objects")
            if r.boxes is not None:
                for box in r.boxes:
                    print(f"Class: {r.names[int(box.cls)]}, Confidence: {box.conf:.2f}")
    else:
        print("Could not capture frame from camera")
    
    cap.release()
else:
    print("Camera not accessible. Model is loaded and ready to use.")
    print("To use the live camera version:")
    print("1. Go to System Preferences > Security & Privacy > Camera")
    print("2. Grant camera access to Terminal or Python")
    print("3. Run 'python3 yolo_detection.py' for live detection")

print("YOLO model test completed!") 