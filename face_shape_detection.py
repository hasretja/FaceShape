from __future__ import print_function
import cv2
import dlib
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
import os
import cv2 as cv
import math
import time
import webcolors
import numpy as np
import binascii
import struct
from PIL import Image
import PIL
import scipy
import scipy.misc
import scipy.cluster

xLocationList=[]
yLocationList=[]
xDiff=[]
yDiff=[]
xlocationDiff = []
ylocationDiff = []
df=DataFrame()
mainDf=DataFrame()
mainData=DataFrame()

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

basedir = os.path.abspath(os.path.dirname(__file__))
faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"

ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 20


def age_gender_detector(frame):
    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        print(genderPreds)
        gender = genderList[genderPreds[0].argmax()]
        print(gender)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()

        age = ageList[agePreds[0].argmax()]
        
        gender=gender+age
        #label = "{},{}".format(gender, age)
        #cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,cv.LINE_AA)
    return gender

start = time.time()
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def deneme(path):
    basedir = os.path.abspath(os.path.dirname(__file__))
    try:
        inputphoto= cv.imread(path)
        outputPhotoGanderAndAge = age_gender_detector(inputphoto)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            basedir+"/shape_predictor_68_face_landmarks.dat")
        img = cv2.imread(path)
        img2 = cv2.imread(path)
        
        #Detecting Edge of image
        Edge = cv2.Canny(img2, 100, 150)
        
        Coords = np.nonzero(Edge)
        
        y = np.min(Coords[0])
        y = y + 20
        h,w,c = img2.shape
        x = int(w/2)-14
        #cv2.circle(img,(x,y), 5, (0,0,255), -1)
        
        color=img2[y,x]
        print(color[0])
        
        requested_colour = (color[2], color[1], color[0])
        actual_name, closest_name = get_colour_name(requested_colour)
        
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            landmarks = predictor(image=gray, box=face)
            for n in range(0, 16):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                xLocationList.append(x)
                yLocationList.append(y)

        print("break")
        df = DataFrame({'Photo Name': path, 'Point X0': xLocationList[0], 'Point Y0': yLocationList[0],
                        'Point X1': xLocationList[1], 'Point Y1': yLocationList[1], 'Point X2': xLocationList[2],
                        'Point Y2': yLocationList[2], 'Point X3': xLocationList[3], 'Point Y3': yLocationList[3],
                        'Point X4': xLocationList[4], 'Point Y4': yLocationList[4], 'Point X5': xLocationList[5],
                        'Point Y5': yLocationList[5], 'Point X6': xLocationList[6], 'Point Y6': yLocationList[6],
                        'Point X7': xLocationList[7], 'Point Y7': yLocationList[7], 'Point X8': xLocationList[8],
                        'Point Y8': yLocationList[8], 'Point X9': xLocationList[9], 'Point Y9': yLocationList[9],
                        'Point X10': xLocationList[10], 'Point Y10': yLocationList[10], 'Point X11': xLocationList[11],
                        'Point Y11': yLocationList[11], 'Point X12': xLocationList[12], 'Point Y12': yLocationList[12],
                        'Point X13': xLocationList[13], 'Point Y13': yLocationList[13], 'Point X14': xLocationList[14],
                        'Point Y14': yLocationList[14], 'Point X15': xLocationList[15], 'Point Y15': yLocationList[15]},
                       index=[0])

        for index, row in df.iterrows():
            xDiff = [row['Point X1'] - row['Point X0'], row['Point X2'] - row['Point X1'],
                     row['Point X3'] - row['Point X2'], row['Point X4'] - row['Point X3'],
                     row['Point X5'] - row['Point X4'], row['Point X6'] - row['Point X5'],
                     row['Point X7'] - row['Point X6']]
            yDiff = [row['Point Y8'] - row['Point Y9'], row['Point Y9'] - row['Point Y10'],
                     row['Point Y10'] - row['Point Y11'], row['Point Y11'] - row['Point Y12'],
                     row['Point Y12'] - row['Point Y13'], row['Point Y13'] - row['Point Y14'],
                     row['Point Y14'] - row['Point Y15']]

        xLoc = 0
        yLoc = 0
        for i in xDiff:
            xLoc = xLoc + i
        for a in yDiff:
            yLoc = yLoc + a

        mainDf = pd.read_excel(basedir+"/mainData.xlsx")
        xlocationDiff.append(xLoc / 7)
        ylocationDiff.append(yLoc / 7)
        try:
            mainData = DataFrame({'X': xlocationDiff, 'Y': ylocationDiff})
        except:
            return "Üzgünüz Yüz şekliniz belirlenemedi"

        # len(mainDf)
        mainDf = mainDf.append(mainData, ignore_index=True)
        kmeans = KMeans(n_clusters=5).fit(mainDf)
        centroids = kmeans.cluster_centers_
        cluster_map = pd.DataFrame()
        cluster_map['data_index'] = mainDf.index.values
        cluster_map['cluster'] = kmeans.labels_
        faceShape = []
        points = []
        cluster_map.to_excel(basedir+'\clusters.xlsx', sheet_name='Sayfa1',
                             index=False)
        
        
        
        
        
        
        
        for a in range(0, 5):
            print(centroids[a])
        for i in range(0, 5):
            points.append(
                list(filter(None, str(centroids[i]).replace("[", "").replace("]", "").replace("  ", "").split())))
            print(points[i])
            try:
                if float(points[i][0]) > 2.94 and float(points[i][0]) < 4.46 and float(points[i][1]) > 5.70 and float(
                        points[i][1]) < 7.00:
                    faceShape.append("Round")
                if float(points[i][0]) > 4.46 and float(points[i][0]) < 5.66 and float(points[i][1]) > 7.60 and float(
                        points[i][1]) < 9.00:
                    faceShape.append("Square")
                if float(points[i][0]) > 7.10 and float(points[i][0]) < 10.93 and float(points[i][1]) > 12.30 and float(
                        points[i][1]) < 17.00:
                    faceShape.append("Oval")
                if float(points[i][0]) > 10.74 and float(points[i][0]) < 12.66 and float(
                        points[i][1]) > 17.36 and float(
                        points[i][1]) < 18.00:
                    faceShape.append("Rectangle")
                if float(points[i][0]) > 5.60 and float(points[i][0]) < 6.26 and float(points[i][1]) > 9.00 and float(
                        points[i][1]) < 10.00:
                    faceShape.append("Triangle")
            except:
                if float((points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) > 2.94 and float(
                        (points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) < 4.46 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) > 5.70 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) < 7.00:
                    faceShape.append("Round")
                if float((points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) > 4.46 and float(
                        (points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) < 5.66 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) > 7.60 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) < 9.00:
                    faceShape.append("Square")
                if float((points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) > 7.10 and float(
                        (points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) < 10.93 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) > 12.30 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) < 17.00:
                    faceShape.append("Oval")
                if float((points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) > 10.74 and float(
                        (points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) < 12.66 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) > 17.36 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) < 18.00:
                    faceShape.append("Rectangle")
                if float((points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) > 5.60 and float(
                        (points[i][0]).split(sep=".")[0] + "." + (points[i][0]).split(sep=".")[1]) < 6.26 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) > 9.00 and float(
                    (points[i][1]).split(sep=".")[0] + "." + (points[i][1]).split(sep=".")[1]) < 10.00:
                    faceShape.append("Triangle")

        return "Your Face Shape :" + faceShape[int(str(cluster_map['cluster'].iloc[-1]))]+": Your Gender And Age :"+outputPhotoGanderAndAge+": Saçının Rengi:"+closest_name
    except:
        return "Bir Hata Oluştu:Bir Hata Oluştu:Bir Hata Oluştu:Bir Hata Oluştu:Bir Hata Oluştu:Bir Hata Oluştu:"