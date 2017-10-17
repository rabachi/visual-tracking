import numpy as np
import cv2
import dlib

if __name__=="__main__":
    
    #Initialize a face cascade using the frontal face haar cascade provided
    #    with the OpenCV2 library
    
    faceCascade = cv2.CascadeClassifier('/Users/romina/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')

    #The deisred output width and height
    OUTPUT_SIZE_WIDTH = 775
    OUTPUT_SIZE_HEIGHT = 600

    #Open the first webcame device
    capture = cv2.VideoCapture(0)

    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    cv2.moveWindow("base-image",0,100)
    cv2.moveWindow("result-image",400,100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()
    rectangleColor = (0,165,255)
    
    tracker = dlib.correlation_tracker()
    trackingFace = 0
    
    while(1):
        #get latest image from webcam
        print("get next image")
        rc, fullSizeBaseImage = capture.read()
        
        baseImage = cv2.resize(fullSizeBaseImage, (320,240))
        
        resultImage = baseImage.copy()
#        cv2.imshow("base-image", baseImage)


        if not trackingFace:
    
            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
#            cv2.imshow("base-image", gray)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    
            print("detecting face...")
    
    #find largest rectangle detected
            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0
    
            for (_x,_y,_w,_h) in faces:
                if _w*_h > maxArea:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)
                    maxArea = w*h
            
            print(maxArea)
    #initialize tracker
            if maxArea>0:
                tracker.start_track(baseImage, dlib.rectangle(x-10,y-20,x+w+10,y+h+20))
                print("tracking face ... ")
                trackingFace = 1
    
        if trackingFace:
            trackingQuality = tracker.update(baseImage)
    
            if trackingQuality >= 8.75:
                tracked_position = tracker.get_position()
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                cv2.rectangle(resultImage, (t_x, t_y),
                          (t_x + t_w , t_y + t_h),
                          rectangleColor ,2)
            else:
                trackingFace = 0

        largeResult = cv2.resize(resultImage,
                             (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

        cv2.imshow("base-image", baseImage)
        cv2.imshow("result-image", resultImage)
        cv2.waitKey(2)
    


    
    
