from ultralytics import YOLO # type: ignore

# Hardcoded path to the image
image_path = "test_image.jpg"

# Load the trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# Run inference
results = model(image_path, conf=0.01)[0]

# Show result
results.show()

# Save result
results.save("predictions/output.jpg")
