
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
#     (h, w) = frame.shape[:2]
    (h,w)=frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the detection
        confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds,faces)


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")
maskNet._make_predict_function()

# initialize the video stream
print("[INFO] starting video stream...")
# vs = cv2.VideoCapture(0)
# statu,frame = vs.read()
# width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
# height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
# size = (width, height)
# print(width, height)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('project_rough.avi', fourcc, 20.0, size)

def compute_distance(midpoints,num):
    dist_a = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1,num):
              if i!=j:
                    dst = dist.euclidean(midpoints[i], midpoints[j])
                    dist_a[i][j]=dst
    return dist_a
# min length for social distancing in mm
thresh = 200
# loop over the frames from the video stream
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
        
    def get_frame(self):   
        statu,frame = self.video.read()
        (h,w)=frame.shape[:2]

        (locs, preds,faces) = detect_and_predict_mask(frame, faceNet, maskNet)
        noss = len(faces)
        print(noss)
        # loop over the detected face locations and their corresponding
        midpoint = []
        dimen=[]
        for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            dimen.append((startX, startY, endX, endY))
            midpoint.append((int((startX+endX)/2), int((startY+endY)/2)))
                
        dist_a = compute_distance(midpoint,noss)
        print(dist_a)
        
        for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
    
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
            font=cv2.FONT_HERSHEY_SIMPLEX
            frame =cv2.putText(frame, label, (startX, startY - 10),
                font, 0.45, color, 2)
            
    #         cv2.circle(frame, (int((startX+endX)/2), int((startY+endY)/2)), 3 , [255,0,0] , -1)
            frame = cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    #         print(mask,withoutMask)
            title = "Mask" if mask > withoutMask else "No Mask"
            if noss>1 and (title=='No Mask'):
                for i in range(noss):
                    for j in range(i,noss):
                        if i!=j & (dist_a[i][j]<=thresh):
    
                            startX, startY, endX, endY=dimen[i]
                            startX2, startY2, endX2, endY2=dimen[j]
    #                         print(startX2, startY2, endX2, endY2)
                            x1,x2=int((startX+endX)/2),int((startX2+endX2)/2)
                            y1,y2 =int((startY+endY)/2),int((startY2+endY2)/2)
                #                         cv2.circle(frame, (int((startX+endX)/2), int((startY+endY)/2)), 5 , [255,0,0] , -1)
                            frame = cv2.line(frame, (x1, y1), (x2,y2), (0,200,200), 2)
                            frame = cv2.putText(frame,"Maintain 6ft apart", (min(x1+10,x2+10), max(y1,y2)),font, 0.7, (200,50,100), 2)
    

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()