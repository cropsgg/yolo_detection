from ultralytics import YOLO
import cv2
import os
import time

# Load your trained model
model_path = "best.pt"
model = YOLO(model_path)

print("=" * 50)
print("YOLO Detection Test & Functionality Check")
print("=" * 50)

# Display model information
print(f"Model loaded: {model_path}")
print(f"Model classes: {list(model.names.values())}")
print(f"Number of classes: {len(model.names)}")
print()

# Test with camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible!")
    exit(1)

print("✅ Camera accessed successfully!")
print("📸 Capturing test images and running detection...")
print()

# Capture and test multiple frames
for i in range(3):
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to capture frame {i+1}")
        continue
    
    print(f"🔍 Processing frame {i+1}...")
    
    # Run detection
    results = model(frame, conf=0.3)  # Lower confidence for more detections
    
    # Annotate and save
    annotated_frame = results[0].plot()
    filename = f"test_detection_{i+1}.jpg"
    cv2.imwrite(filename, annotated_frame)
    
    # Display results
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            print(f"  📋 Detected {len(r.boxes)} objects:")
            for j, box in enumerate(r.boxes):
                class_name = r.names[int(box.cls)]
                confidence = float(box.conf)
                print(f"    {j+1}. {class_name} (confidence: {confidence:.2f})")
        else:
            print(f"  📋 No objects detected in frame {i+1}")
    
    print(f"  💾 Saved as: {filename}")
    print()
    
    time.sleep(1)  # Wait 1 second between captures

cap.release()

print("=" * 50)
print("✅ Test completed! Check the saved images:")
print("   • test_detection_1.jpg")
print("   • test_detection_2.jpg") 
print("   • test_detection_3.jpg")
print()
print("🎥 Your live detection is running in the background.")
print("   Look for the 'YOLOv8 Phone Detection' window.")
print("   Press 'q' in that window to stop live detection.")
print("=" * 50) 