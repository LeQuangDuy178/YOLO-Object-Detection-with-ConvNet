##--------------------------------------------- YOLOv11 Test --------------------------------------------------
# Test YOLO v11 Inference

from ultralytics import YOLO

# Create the YOLO model from scratch (Untrained)
#model = YOLO(Path = "yolo11n.yaml")

# Load the pre-trained YOLO model (.pt for PyTorch, .tflite for TensorFlow lite)
model = YOLO(model = "yolo11n.pt")

# Train the model on the COCO 8 'coco8.yaml' dataset for 3 epochs
trained_results = model.train(data="coco8.yaml", epochs=3, imgsz=640, device="cpu") 

# Evaluate and validate the model's performance on the validation dataset
metrics = model.val()

# Perform object detection inference on an image using the model
results = model(source = "images/0024.jpg")
results[0].show()

# Export the model to ONNX (or tflite) format
success = model.export(format = "tflite")