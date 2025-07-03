import cv2
import imutils
import numpy as np

# Use sample from https://github.com/opencv/opencv/blob/master/samples/python/tracker.py
# as param setting hint

# 'csrt': cv2.TrackerCSRT,
# 'kcf' : cv2.TrackerKCF,
# 'boosting' : cv2.TrackerBoosting,
# 'tld': cv2.TrackerTLD,
# 'medianflow': cv2.TrackerMedianFlow,
# 'mosse':cv2.TrackerMOSSE,
TrDict = {
     'mil': cv2.TrackerMIL,
     'GORETURN': cv2.TrackerGOTURN,
     'Vit': cv2.TrackerVit,
     'Nano': cv2.TrackerNano,
     'none': None}

trackAlgorithm = 'Nano'
#trackAlgorithm = 'MIL'

# Nano models from https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

if trackAlgorithm == 'Nano':
    params = cv2.TrackerNano.Params()
    params.backbone = "data/nanotrack_backbone_sim.onnx"
    params.neckhead = "data/nanotrack_head_sim.onnx"
    tracker = cv2.TrackerNano.create(parameters=params)
elif trackAlgorithm == 'MIL':
    params = cv2.TrackerMIL.Params()
    tracker = cv2.TrackerMIL.create(parameters=params)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ok, frame = video.read()
if not ok:
    print(f"Video not available")
    quit(1)
bbox = cv2.selectROI(frame)
if (np.asarray(bbox) == 0).all():
    print(f"ROI not selected")
    quit(2)
tracker.init(frame,bbox)
while True:
   ok, frame = video.read()
   if not ok:
        break
   ok, bbox = tracker.update(frame)
   if ok:
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
   else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
   cv2.imshow('Tracking',frame)
   cv2.imshow('Tracking',frame)
   if cv2.waitKey(1) & 0XFF==27:
        break
video.release()
cv2.destroyAllWindows()
