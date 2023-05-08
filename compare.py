from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


# class Tracker:
#     tracker = None
#     encoder = None
#     tracks = None
#     last_id = 0
#
#     def __init__(self, bbox):
#         self.track_id = Track.last_id
#         Track.last_id += 1
#         self.bbox = bbox
#
#     def __init__(self):
#         max_cosine_distance = 0.4
#         nn_budget = None
#
#         encoder_model_filename = 'model_data/mars-small128.pb'
#
#         metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#         self.tracker = DeepSortTracker(metric)
#         self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
#
#     def update(self, frame, detections):
#
#         bboxes = np.asarray([d[:-1] for d in detections])
#         bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
#         scores = [d[-1] for d in detections]
#
#         features = self.encoder(frame, bboxes)
#
#         dets = []
#         for bbox_id, bbox in enumerate(bboxes):
#             dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
#
#         self.tracker.predict()
#         self.tracker.update(dets)
#         self.update_tracks()
#
#     def update_tracks(self):
#         tracks = []
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             bbox = track.to_tlbr()
#
#             id = track.track_id
#
#             tracks.append(Track(id, bbox))
#
#
#         self.tracks = tracks
#
#
# class Track:
#     track_id = None
#     bbox = None
#
#     def __init__(self, id, bbox):
#         self.track_id = id
#         self.bbox = bbox

class Tracker:
    tracker = None
    encoder = None
    tracks = None
    max_track_id = 0 # add this variable

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = 'model_data/mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):
        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id if track.track_id is not None else self.max_track_id + 1 # modify this line
            tracks.append(Track(id, bbox))
            if id > self.max_track_id: # add this line
                self.max_track_id = id # add this line
        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox




# class Track:
#     track_id = None
#     bbox = None
#
#     def __init__(self, id, bbox):
#         self.track_id =id
#         self.bbox = bbox
#
# class Tracker:
#     tracker = None
#     encoder = None
#     tracks = None
#     max_track_id = 0 # add this variable
#
#
#     def __init__(self):
#         max_cosine_distance = 0.4
#         nn_budget = None
#         encoder_model_filename = 'model_data/mars-small128.pb'
#         metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#         self.tracker = DeepSortTracker(metric)
#         self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
#
#     def update(self, frame, detections):
#         bboxes = np.asarray([d[:-1] for d in detections])
#         bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
#         scores = [d[-1] for d in detections]
#         features = self.encoder(frame, bboxes)
#         dets = []
#         for bbox_id, bbox in enumerate(bboxes):
#             dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
#         self.tracker.predict()
#         self.tracker.update(dets)
#         self.update_tracks()
#
#     def update_tracks(self):
#         tracks = []
#         for track in self.tracker.tracks:
#             # print(self.tracker.tracks)
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             bbox = track.to_tlbr()
#             if track.track_id is not None:
#                 print("dakhlt")
#                 id=track.track_id+1
#                 print(id)
#             else:
#                 print("entered")
#                 id=track.track_id
#
#             # id = track.track_id if track.track_id is not None else self.max_track_id + 1 # modify this line
#             tracks.append(Track(id, bbox))
#
#             if int(id) > int(self.max_track_id): # add this line
#                 self.max_track_id = track.track_id # add this line
#
#         self.tracks = tracks
#
#
#
