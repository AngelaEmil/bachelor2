from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
# from tracker import Tracker
import random
from imutils.object_detection import non_max_suppression
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching

max_cosine_distance = 0.4
nn_budget = None
encoder_model_filename = 'model_data/mars-small128.pb'
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = DeepSortTracker(metric)
model = YOLO("yolov8n.pt", "v8")
cap = cv2.VideoCapture("slow.mp4")
# tracker=Tracker()

colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(10)]

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (1000, 700))
    if frame is None:
        continue
    results = model.predict(frame, conf=0.5)
    length= len(results[0].numpy().boxes)

    car = 0
    truck=0
    bus=0
    motorcycle=0




    for result in results:

        detections=[]
        for r in result.boxes.data.tolist():
            # if (results[0].numpy().boxes[r].cls) == 2:

                x1,y1,x2,y2, score, class_id=r
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                class_id = int(class_id)
                if (class_id==2 or class_id==3 or class_id==5 or class_id==7):
                    detections.append([x1, y1, x2, y2,score])
                    # detections = non_max_suppression(detections, probs=None, overlapThresh=0.3)
                    if (class_id==2):
                        car+=1
                    elif (class_id==5):
                        bus+=1
                    elif (class_id==3):
                        motorcycle+=1
                    elif (class_id==7):
                        truck+=1
                    tracker.update(detections)
                    for track in tracker.tracks:

                        track_id=track.track_id
                        # print(track_id)
                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(colors[track_id%len(colors)]),3)
                        cv2.putText(frame, f'ID:{track_id}', (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    # (frame,text,start,font(0-7),fontScale,color,thickness,lineType)
    cv2.putText(frame, f'Cars: {car}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Buses: {bus}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Trucks: {truck}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Motorcycles: {motorcycle}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("image", frame)
    key =cv2.waitKey(1)
    if key == 27:
        break
