# Import the Open-CV extra functionalities
import cv2
import os
import json
import numpy as np

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the list of furniture items from furniture_items.txt
furniture_items_file = os.path.join(current_dir, "furniture_items.txt")
with open(furniture_items_file, "rt") as f:
    furniture_items = [line.strip() for line in f.readlines()]

# Our chosen reference object to estimate the size of other objects
ref_obj = {"Name": "chair",
           "Width": 50.0,
           "Height": 80.0}

# This is to pull the information about what each object is called
classNames = []
classFile = os.path.join(current_dir, "coco.names")  # Use absolute path
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
configPath = os.path.join(current_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")  # Use absolute path
weightsPath = os.path.join(current_dir, "frozen_inference_graph.pb")  # Use absolute path

# This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    scale_width = scale_height = 1  # Initialize scale factors
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                # Estimate size of the object
                if className == ref_obj["Name"]:  # Replace with your reference object class name
                    ref_width = ref_obj["Width"]  # Real-world width of the reference object in cm
                    ref_height = ref_obj["Height"]  # Real-world height of the reference object in cm
                    pixel_width = box[2]
                    pixel_height = box[3]
                    scale_width = ref_width / pixel_width
                    scale_height = ref_height / pixel_height
                    objectInfo[-1].extend([ref_width, ref_height])
                else:
                    pixel_width = box[2]
                    pixel_height = box[3]
                    estimated_width = pixel_width * scale_width
                    estimated_height = pixel_height * scale_height
                    objectInfo[-1].extend([estimated_width, estimated_height])
                # Draw the bounding box and labels
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    if className != ref_obj["Name"]:
                        cv2.putText(img, f"Width: {estimated_width:.2f} cm", (box[0], box[1] - 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f"Height: {estimated_height:.2f} cm", (box[0], box[1] - 50),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


def processPhotos(folder_path, output_folder):
    """
    Processes all images in the specified folder and saves the results to the output folder.
    Additionally, creates a results.json file with details of detected objects.

    Args:
        folder_path (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images and results.json will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    results = []  # List to store results for all processed images

    # Empty the output folder
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for both .jpg and .png
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue

            # Process the image using furniture_items
            result, objectInfo = getObjects(img, 0.45, 0.2, objects=furniture_items)
            print(f"Processed {filename}: {objectInfo}")

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)

            # Add the results for this image to the results list
            results.append({
                "image": filename,
                "objects": [
                    {
                        "class": obj[1],
                        "box": [int(coord) for coord in obj[0]] if isinstance(obj[0], (list, tuple, np.ndarray)) else obj[0],  # Convert to list of ints
                        "width": float(obj[2]) if len(obj) > 2 else None,  # Convert to float
                        "height": float(obj[3]) if len(obj) > 3 else None  # Convert to float
                    }
                    for obj in objectInfo
                ]
            })

    # Delete an existing results.json file if it exists
    results_file = os.path.join(output_folder, "results.json")
    if os.path.exists(results_file):
        os.remove(results_file)
    
    # Save the results to a JSON file
    results_file = os.path.join(output_folder, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "../photos"  # Path to the folder containing images
    output_folder = "../output"  # Path to save processed images
    processPhotos(folder_path, output_folder)
