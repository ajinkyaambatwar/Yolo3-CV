import cv2
import numpy as np
import argparse

#INITIALIZING PARAMETERS
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INP_WIDTH = 416
INP_HEIGHT = 416

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()


#LOADING CLASSES NAMES
coco_classes = 'coco_classes.txt'
classes = None
with open(coco_classes, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

#GET THE NAMES OF THE OUTPUT LAYERS
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, confid, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))
    label = '%.2f' %confid
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

#POSTPROCESSING
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    #GET THE BOX EDGES AND DIMENSION 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence >  CONFIDENCE_THRESHOLD:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(centerX - width/2)
                top = int(centerY - height/2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    #PERFORM NON-MAXIMUM SUPRESSION TO ELIMAINATE THE OVERLAPPING BOXES
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i],confidences[i], left, top, left+width, top+height)
    
if(args.video):
    outputFile = "output/yolo_output.avi"
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)


vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 
                                30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('DOne!!')
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255, (INP_WIDTH,INP_HEIGHT), [0,0,0], 1, crop=False)

    #SETS THE INPUTS FOR THE DARKNET NETWORK
    net.setInput(blob)

    #RUNS THE FORWARD PASS THROUGH THE NETWORK
    outs = net.forward(getOutputsNames(net))

    #POSTPROCESS
    postprocess(frame, outs)
    
    #EFFICIENCY INFO
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t/cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", frame)
    vid_writer.write(frame.astype(np.uint8))

