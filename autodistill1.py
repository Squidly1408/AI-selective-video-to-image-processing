import os
from ultralytics import YOLO

# Define your classes and their descriptions
classes = {
    "car": "A motor vehicle with four wheels, typically powered by an internal combustion engine.",
    "bus": "A large motor vehicle carrying passengers by road, typically one serving the public on a fixed route.",
    "truck": "A motor vehicle designed to transport cargo.",
    'short vehicle':'less then 3.2 meters and 2 axles',
    'short vehicle with trailer':'groups of wheels 3 axles equal to three or four or five' ,
    'two axle truck or bus':'greater then 3.2 meters and axles two',
    'three axle truck or bus':'axles equal to three and groups of tires equal to two',
    'four axle truck':'axles equal to four and groups of tires equal to two',
    'three axle articulate':'greater then 3.2 meters three axles and three groups of tires',
    'four axle articulate':'four axles and 3 groups of tires',
    'five axle articulate':'5 axles and three groups of tires',
    'six axle articulate':'6 axles and three to 5 groups of tires',
    'b double':'4 groups of tires and 6 axles',
    'double road train':'groups of tires equal to 5 or 6 and axles greater than 6',
    'triple road train':'groups greater than 6 and axles greater than 6' 
    # Add more classes as needed
}

# Load YOLOv8 model (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO('yolov8n.pt')  # Replace with your desired YOLOv8 model

# Directory containing your images
image_folder = "images"

# Function to perform auto annotation based on the provided classes
def auto_annotate(model, image_folder, classes):
    # Loop through each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path):
            # Perform inference using YOLOv8
            results = model(image_path)
            annotations = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    if class_name in classes:
                        annotations.append((class_name, box.xyxy.tolist(), box.conf.item()))

            # Save annotations or perform further processing
            if annotations:
                save_annotations(image_path, annotations)
                print(f"Annotated image {image_name} with {len(annotations)} annotations.")

# Function to save annotations (this can be customized based on your needs)
def save_annotations(image_path, annotations):
    annotation_path = image_path + ".txt"
    with open(annotation_path, 'w') as f:
        for class_name, bbox, confidence in annotations:
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{class_name} {bbox_str} {confidence}\n")

# Perform auto annotation
auto_annotate(model, image_folder, classes)
