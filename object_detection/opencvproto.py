# Import the Open-CV extra functionalities
import cv2

# Our chosen reference object to estimate the size of other objects
ref_obj = {"Name": "chair",
           "Width": 50.0,
           "Height": 80.0}

# This is to pull the information about what each object is called
classNames = []
classFile = "./coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
configPath = "./ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "./frozen_inference_graph.pb"

# This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    # Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below
    # print(classIds, bbox)
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

if __name__ == "__main__":

    cap = cv2.VideoCapture("test.mp4")
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10, 70)

    # Below is the never ending loop that determines what will happen when an object is identified.
    while True:
        success, img = cap.read()
        if not success:
            break
        # Below provides a huge amount of control. the 0.45 number is the threshold number, the 0.2 number is the nms number)
        result, objectInfo = getObjects(img, 0.45, 0.2, objects=["chair"])
        print(objectInfo)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()