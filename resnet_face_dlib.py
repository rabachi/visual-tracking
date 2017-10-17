import numpy as np
import argparse
import os
import sys
sys.path.append('/home/arrybn/build/opencv/lib')
import cv2
import dlib

from cv2 import dnn

inWidth = 300
inHeight = 300
confThreshold = 0.7

prototxt = 'face_detector/deploy.prototxt'
caffemodel = 'face_detector/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

if __name__ == '__main__':
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("result-image",300,300)
    cv2.startWindowThread()
    
    tracker = dlib.correlation_tracker()
    trackingFace = 0
    while True:
        ret, frame = cap.read()
#         frame = cv2.resize(frame2,(300,300))
        cols = frame.shape[1]
        rows = frame.shape[0]

        net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)),
                                       1.0, (inWidth, inHeight), (104., 177., 123.)))
        detections = net.forward()

#        perf_stats = net.getPerfProfile()
#
#        print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))


        if not trackingFace:
            print(detections.shape)
            for i in range(detections.shape[2]):
#            i_arg = np.argmax(detections,axis=2)
#            i_max = np.amax(detections,axis=2)
#            print(i_max)
#            print(i_arg.shape)
#            print(np.argmax(i_max))
#            i = i_arg[:,:,np.argmax(i_max)]
#            print(i)
                confidence = detections[0, 0, i, 2]
                if confidence > confThreshold:
                    xLeftBottom = int(detections[0, 0, i, 3]* cols)
                    yLeftBottom = int(detections[0, 0, i, 4]* rows)
                    xRightTop = int(detections[0, 0, i, 5]* cols)
                    yRightTop = int(detections[0, 0, i, 6]* rows)

    #                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))
    #                    label = "face: %.4f" % confidence
    #                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    #                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),(xLeftBottom + labelSize[0], yLeftBottom + baseLine),(255, 255, 255), cv.FILLED)
    #                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
    #                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    tracker.start_track(frame, dlib.rectangle(xLeftBottom,yLeftBottom,xRightTop,yRightTop))
                    print("tracking face ... ")
                    trackingFace = 1
    

        if trackingFace:
            trackingQuality = tracker.update(frame)
        
            if trackingQuality >= 8.75:
                tracked_position = tracker.get_position()
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                cv2.rectangle(frame, (t_x, t_y),
                              (t_x + t_w , t_y + t_h),
                              (0, 255, 0) ,2)
            else:
                trackingFace = 0
    
    
    

        cv2.imshow("result-image", frame)
        if cv2.waitKey(1) != -1:
            break
