# lightweight-face-detection-using-Dlib-and-python
small API for using Dlib in python to create face detection and facial landmark detection system.
achieve real-time performance on edge devices and CPU with high tolrance for posed faces.

## How to use 
clone project, copy files to your project and you are good to go. 
example :  

1- import wrapper class
    
    from detector_light_dlib import FaceDetectorLightDlib
    
2- create instance and load the facial landmark and face detection models 

    detection_system = FaceDetectorLightDlib()
    detection_system.load()

3- call your detector on images or video frames 

    bounding_boxes, facial_points = detection_system.detect(image)



## excpected ouput 
a numpy array of bounding boxes for faces and numpy array of n element each containes x,y coordinates of 5 facial points (two for each eye and one for the nose) 

have fun !!
