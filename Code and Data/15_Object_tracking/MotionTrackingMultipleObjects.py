import cv2
#import imutils
import numpy as np

# Looks that cv2.MultiTracker is no available in recent versions.
# Needs to make list of trackers for same frame

trackAlgorithm = 'Nano'
#trackAlgorithm = 'MIL'

# Nano models from https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

object_count_to_track = 4

trackers = []
for i in range(object_count_to_track):
    if trackAlgorithm == 'Nano':
        params = cv2.TrackerNano.Params()
        params.backbone = "data/nanotrack_backbone_sim.onnx"
        params.neckhead = "data/nanotrack_head_sim.onnx"
        tracker = cv2.TrackerNano.create(parameters=params)
    elif trackAlgorithm == 'MIL':
        params = cv2.TrackerMIL.Params()
        tracker = cv2.TrackerMIL.create(parameters=params)
    else:
        print(f"Tracker name {trackAlgorithm} is not implemented")
        quit(1)
    trackers.append(tracker)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ok, frame = video.read()
if not ok:
    print(f"Video not available")
    quit(1)


for i in range(object_count_to_track):
    print(f"There are {object_count_to_track-i} objects to select")
    #cv2.imshow('Frame', frame)
    bbi = cv2.selectROI('Frame', frame)
    trackers[i].init(frame, bbi)
cv2.destroyWindow('Frame')

while True:
    ok, frame = video.read()
    if not ok:
        break
    for i in range(object_count_to_track):
        ok, bbox = trackers[i].update(frame)
        if ok:
            (x,y,w,h)=[int(v) for v in bbox]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
        else:
            cv2.putText(frame, f"Error in {i}",(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Tracking',frame)
    cv2.imshow('Tracking',frame)
    if cv2.waitKey(1) & 0XFF==27:
        break
video.release()
cv2.destroyAllWindows()