from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
# from deep_sort.deep_sort import nn_matching
# from deep_sort.deep_sort.tracker import Tracker
from tracker import Tracker
from colorTry import color_detection
import random


#load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")
# img = cv2.imread('datasets/bus.jpg')
cap = cv2.VideoCapture("slow.mp4")
# initialize tracker

max_cosine_distance = 0.4
nn_budget = None
colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(10)]


# encoder_model_filename = 'model_data/mars-small128.pb'

# metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# tracker = Tracker(metric)
tracker = Tracker()
total=0
id =0

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (1000, 700))
    if frame is None:
        continue
    result = model.predict(frame, conf=0.3)
    length= len(result[0].numpy().boxes)

    car = 0
    truck=0
    bus=0
    motorcycle=0




    for x in range(length):

        # detections.append([x1, y1, x2, y2, score])

        if (result[0].numpy().boxes[x].cls) == 2:
            car = car + 1
            detections=[]
            x1 = int(result[0].numpy().boxes[x].xyxy[0][0])
            y1 = int(result[0].numpy().boxes[x].xyxy[0][1])
            x2 = int(result[0].numpy().boxes[x].xyxy[0][2])
            y2 = int(result[0].numpy().boxes[x].xyxy[0][3])
            class_id = int(result[0].numpy().boxes[x].cls)
            score = result[0].numpy().boxes[x].conf
            detections.append([x1, y1, x2, y2, score])
            tracker.update(frame, detections)
            # tracker.update(detections)
            # for track in tracker.tracks:
            #     bbox = track.bbox
            #     track_id = track.track_id
            #     total=track_id
            #     id=track_id
            for track in tracker.tracks:
                # print("Object with track_id {} detected".format(track.track_id))
                id = track.track_id
            rect = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            roi=frame[y1:y2,x1:x2]
            print(roi.shape)
            # print(rect.shape)

            cv2.putText(rect, f'ID:{id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            color= color_detection(roi)
            cv2.putText(frame, color, (x2-10, y2), 0, 1, (0, 0, 0), 2)

        elif (result[0].numpy().boxes[x].cls) == 5:
            bus = bus + 1
            detections = []
            x1 = int(result[0].numpy().boxes[x].xyxy[0][0])
            y1 = int(result[0].numpy().boxes[x].xyxy[0][1])
            x2 = int(result[0].numpy().boxes[x].xyxy[0][2])
            y2 = int(result[0].numpy().boxes[x].xyxy[0][3])
            class_id = int(result[0].numpy().boxes[x].cls)
            score = result[0].numpy().boxes[x].conf
            detections.append([x1, y1, x2, y2, score])
            tracker.update(frame, detections)
            # tracker.update( detections)
            # for track in tracker.tracks:
            #     bbox = track.bbox
            #     track_id = track.track_id
            #     total = track_id
            #     id=track_id
            for track in tracker.tracks:
                # print("Object with track_id {} detected".format(track.track_id))
                id=track.track_id
            cv2.putText(frame, f'ID:{id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            rect = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            roi=frame[y1:y2,x1:x2]

            color = color_detection(roi)
            cv2.putText(frame, color, (x2-10, y2-10), 0, 1, (0, 0, 0), 2)

        elif (result[0].numpy().boxes[x].cls) == 3:
            motorcycle = motorcycle + 1
            detections = []
            x1 = int(result[0].numpy().boxes[x].xyxy[0][0])
            y1 = int(result[0].numpy().boxes[x].xyxy[0][1])
            x2 = int(result[0].numpy().boxes[x].xyxy[0][2])
            y2 = int(result[0].numpy().boxes[x].xyxy[0][3])
            class_id = int(result[0].numpy().boxes[x].cls)
            score = result[0].numpy().boxes[x].conf
            detections.append([x1, y1, x2, y2, score])
            tracker.update(frame, detections)
            # tracker.update(detections)
            # for track in tracker.tracks:
            #     bbox = track.bbox
            #     track_id = track.track_id
            #     total = track_id
            #     id=track_id
            for track in tracker.tracks:
                # print("Object with track_id {} detected".format(track.track_id))
                id = track.track_id
            cv2.putText(frame, f'ID:{id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            rect = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            roi=frame[y1:y2,x1:x2]

            color = color_detection(roi)
            cv2.putText(frame, color, (x2-10, y2-10), 0, 1, (0, 0, 0), 2)

        elif (result[0].numpy().boxes[x].cls) == 7:
            truck = truck + 1
            detections = []
            x1 = int(result[0].numpy().boxes[x].xyxy[0][0])
            y1 = int(result[0].numpy().boxes[x].xyxy[0][1])
            x2 = int(result[0].numpy().boxes[x].xyxy[0][2])
            y2 = int(result[0].numpy().boxes[x].xyxy[0][3])
            class_id = int(result[0].numpy().boxes[x].cls)
            score = result[0].numpy().boxes[x].conf
            detections.append([x1, y1, x2, y2, score])
            tracker.update(frame, detections)
            # tracker.update(detections)
            # for track in tracker.tracks:
            #     bbox = track.bbox
            #     track_id = track.track_id
            #     total = track_id
            #     id=track_id
            for track in tracker.tracks:
                # print("Object with track_id {} detected".format(track.track_id))
                id = track.track_id
            cv2.putText(frame, f'ID:{id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            rect = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            roi=frame[y1:y2,x1:x2]

            color = color_detection(roi)
            cv2.putText(frame, color, (x2-10, y2-10), 0, 1, (0, 0, 0), 2)

    # (frame,text,start,font(0-7),fontScale,color,thickness,lineType)
    cv2.putText(frame, f'Cars: {car}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Buses: {bus}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Trucks: {truck}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Motorcycles: {motorcycle}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Total: {total}', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("image", frame)
    key =cv2.waitKey(1)
    if key == 27:
        break
