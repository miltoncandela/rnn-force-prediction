import cv2 as cv
import matplotlib.pyplot as plt
import os

print(os.getcwd())

net = cv.dnn.readNetFromTensorflow('course1/graph_opt.pb')

imWidth = 368
imHeight = 368
thr = 0.2

BODY_PARTS = {'Nose':0, 'Neck':1, 'RShoulder':2, 'RElbow': 3, 'RWrist' : 4,
               'LShoulder': 5, 'LElbow': 6, 'LWrist': 7, 'RHip': 8, 'RKnee': 9,
               'RAnkle': 10, 'LHip': 11, 'LKnee': 12, 'LAnkle':13, 'REye':14,
               'LEye': 15, 'REar': 16, 'LEar': 17, 'Background': 18}

POSE_PARTS = [ ['Neck', 'RShoulder'], ['Neck', 'LShoulder'], ['RShoulder', 'RElbow'],
               ['RElbow', 'RWrist'], ['LShoulder', 'LElbow'], ['LElbow', 'LWrist'],
               ['Neck', 'RHip'], ['RHip', 'RKnee'], ['RKnee', 'RAnkle'], ['Neck', 'LHip'],
               ['LHip', 'LKnee'], ['LKnee', 'LAnkle'], ['Neck', 'Nose'], ['Nose', 'REye'],
               ['REye', 'REar'], ['Nose', 'LEye'], ['LEye', 'LEar'] ]

def leer_imagen():
    img = cv.imread('course1/img.jpg')
    cv.imshow('Cat', img)
    cv.waitKey(0)

def leer_video():
    capture = cv.VideoCapture('videosBiomec/tirolibre_estudiante.mp4')
    cv.waitKey(0)

    while True:
        isTrue, frame = capture.read()
        cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()

def pose_estimation(frame):
    frameWidth = frame.shape[1]
    framwHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (imWidth, imHeight),
                 (127.5, 127.5, 127.5), crop = False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])

    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0] / out.shape[3])
        y = (framwHeight * point[1] / out.shape[2])

        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PARTS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360,  (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    return frame

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[2] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions)

#nombres = ['nerda1', 'nerda2', 'sansan1', 'sansan2',
#           'sansan3', 'sansan4', 'sansan5']

#nombres_bien = ['nerda1', 'nerda2', 'sansan3', 'sansan4']
#img = cv.imread('fotosNormales/sansan3.jpeg')
#print(img)
#img_rescaled = rescaleFrame(img)
#cv.imshow('cat', img_rescaled)
#cv.waitKey(0)

#for nombre in nombres_bien:
#    img = cv.imread('fotosNormales/' + nombre + '.jpeg')
#    estimated_image = pose_estimation(img)
#    cv.imshow(nombre, estimated_image)
#    cv.waitKey(0)

#cap = cv.VideoCapture('videosNormales/nerdasalsa1.mp4')
cap = cv.VideoCapture('videosBiomec/tirolibre_estudiante.mp4')
cap.set(cv.CAP_PROP_FPS, 10)
cap.set(3, 800)
cap.set(4, 800)

if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open video')

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    frameWidth = frame.shape[1]
    framwHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (imWidth, imHeight),
                 (127.5, 127.5, 127.5), crop = False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])

    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0] / out.shape[3])
        y = (framwHeight * point[1] / out.shape[2])

        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PARTS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360,  (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    cv.imshow('Pose estimation', frame)
