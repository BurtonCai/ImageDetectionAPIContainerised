# Imported necessary packages for image recognition and parsing. 
import numpy as np
import io
import sys
import time
import cv2
import os
import json
import base64
from PIL import Image 
from flask import Flask, request, Response, jsonify


# Initialize Application through calling Flask.
app = Flask(__name__) 

# Construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1 


def get_labels(labels_path):
    # Load the COCO class labels our YOLO model was trained on
    lpath=os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)    
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # Derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def do_prediction(image,net,LABELS, image_id):

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and class IDs. Created object list for detection method. 
    boxes = []
    confidences = []
    classIDs = []
    objects = {}

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # Stores variables in appropriate data structure and reiteratively appends them based off pre-given code. 
    if len(idxs) > 0:
        for i in idxs.flatten():
            object_details = {       
                'label': LABELS[classIDs[i]],
                'accuracy': confidences[i],
                'rectangle': {
                    'height' : boxes[i][0],
                    'left': boxes[i][1],
                    'top': boxes[i][2],
                    'width': boxes[i][3],
                }}
            result.setdefault('objects', []).append(object_details)
    objects['id'] = image_id
    return jsonify(objects)

#Remove need for yolo_tiny_configs as an argument to boot the web service. 
yolo_path = str("yolo_tiny_configs")

# Removed input requirement for arguments. 

# Required YoloV3 files and their locations. 
labelsPath= "coco.names"
cfgpath= "yolov3-tiny.cfg"
wpath= "yolov3-tiny.weights"

Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)


#Implementation of script as a Web API using Post Methods and Flask. 
@app.route('/api/object_detection', methods=['POST'])
def main():
    json_image = json.loads(request.json)
    image_id = json_image['id']
    base_image = base64.b64decode(json_image['image'])
    image_file = Image.open(io.BytesIO(base_image))
    np_img = np.array(image_file)
    image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    nets = load_model(CFG, Weights)
    jsonObjects = do_prediction(image, nets, Lables, image_id)
    return jsonObjects
    
if __name__ == "__main__":
    app.run(debug = True, threaded = True, host = '0.0.0.0')