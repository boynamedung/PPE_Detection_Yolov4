import datetime
import cv2
import numpy as np
import time

def boxformat(frame, classnum, counter, id ,x ,y, h, w, nameclass, font, color, size, thickness):
    if ((classNames[classIds[classnum]].upper())==str(nameclass)):
        counter += 1
        id = "ID: " + str(counter)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, f'{str(id)}{classNames[classIds[classnum]].upper()}', (x, y - 5), font, size, color, thickness)

cap = cv2.VideoCapture(0)
whT = 320
confThershold = 0.5
nmsThreshold = 0.3
counter_khongbaoho = 0
counter_nguoi = 0
counter_baohochuan = 0
counter_baohokhongchuan = 0
id_khongbaoho = ""
id_baohochuan = ""
id_baohokhongchuan = ""
classNames = []
classesFile = 'D:/Data old/send/yolo.names'
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'D:/Data old/send/yolov4-tiny-custom.cfg.txt' 
modelWeight = 'D:/Data old/send/yolov4-tiny-custom_best.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
if (cv2.cuda.getCudaEnabledDeviceCount()) > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
while True:
    num_frame = 2;
    start = time.time()
    for i in range(0, num_frame):
        success, img, = cap.read()
    end = time.time()
    seconds = end - start
    fps = round(num_frame/seconds,0)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThershold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThershold, nmsThreshold)
    local_time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    local_time_filename = str(datetime.datetime.now().strftime("%Y %m %d_%H %M %S"))
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if ((classNames[classIds[i]].upper()) != 0):
            counter_nguoi = counter_nguoi + 1
        boxformat(img, i, counter_khongbaoho, id_khongbaoho, x, y, h, w, "KHONGBAOHO", cv2.FONT_HERSHEY_COMPLEX, (0, 0, 255), 0.6 , 2)
        boxformat(img, i, counter_baohochuan, id_baohochuan, x, y, h, w, "BAOHOCHUAN", cv2.FONT_HERSHEY_COMPLEX, (0, 255, 0), 0.6 , 2)
        boxformat(img, i, counter_baohokhongchuan, id_baohokhongchuan, x, y, h, w, "BAOHOKHONGCHUAN", cv2.FONT_HERSHEY_COMPLEX, (0, 102, 255), 0.6, 2)
    cv2.putText(img, "So cong nhan : " + str(counter_nguoi) + " at Area 2A, Samsung INC", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, local_time, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "FPS: ", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 0), 2)
    cv2.putText(img, str(fps), (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 0), 2)
    counter_khongbaoho = 0
    counter_nguoi = 0
    counter_baohochuan = 0
    counter_baohokhongchuan = 0
    cv2.imshow("window", img)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



