'''
    face detection using light dlib models 
'''

import os
import dlib
import numpy as np
import cv2
from .detector_base import FaceDetector
from imutils import face_utils
import time
class FaceDetectorLightDlib(FaceDetector):

    NAME = 'detector_dlib'

    __slots__=['detector','predictor']
    def __init__(self):
        super(FaceDetectorLightDlib, self).__init__()
        self.detector = None
        self.predictor = None
        self.is_loaded = False

    def name(self):
        '''
        returns the name of the model 
        
        '''
        return FaceDetectorLightDlib.NAME

    def detect(self, npimg,clean=False):      
        '''
         perform dlib face detection in image
         
         Args: 
             npimg (numpy Array) : the image to detect
             clean (boolean) : flag to indicate if you wish to clean the loaded model and configuration or not 
        
         Returns : 
             boxes (numpy array) : the bounding boxes of the faces in the image of shape (n,x,y,w,h,convidance)
             points (numpy array) : the coordinations of the face feature points of shape (n,5,2)
         '''
        if not self.is_loaded:
            self.load()
            self.is_loaded=True
            
        gray = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        points=[]
        ptns=[]
        for rect in rects:
            ptns = self.predictor(gray, rect)        
            points.append(ptns)
        if clean :
             self.clean()
        return self.parse_output(rects,points)
      
  
    
    def parse_output(self,boxes,points):
        '''
        helper function to parse ouput of the detector to fit in the pipline
        
        Args:
           boxes (numpy array) : the faces bounding boxes of the image 
           points (numpy array) : the coordinations of the face feature points
           
        Returns:
            boxes (numpy array) : the faces bounding boxes of the image  in shape (n,x,y,w,h,confidance)
            points (numpy array) : the coordinations of the face feature points in shape (n,2,5)
        '''
        ret =[]
        for ptn in points :
            p = face_utils.shape_to_np(ptn)
            ret.append(p)
        points=ret
        ret=[]
        for box in boxes: 
            bbox= face_utils.rect_to_bb(box)
            bbox=np.asarray(bbox)
            bbox=np.append(bbox, 1) 
            bbox[2]=bbox[2]+bbox[0]
            bbox[3]=bbox[3]+bbox[1]
            ret.append(bbox)
            
        boxes=ret
        if len(points) >1:
            points= np.squeeze(points)
            boxes=np.squeeze(boxes)
         
        else:
            points = np.asarray(points)
            boxes= np.asarray(boxes)

        return boxes,points
    
    def load(self):
        '''
        load the dlib model to prepare for use
        
        Returns:
            True if the model was loaded successfully and False elsewise
        '''
        path =os.path.join(os.path.dirname(os.path.realpath(__file__)),'model')
        path = os.path.join(path,'shape_predictor_5_face_landmarks.dat')
        print(path)
        self.predictor = dlib.shape_predictor(path)
        self.detector = dlib.get_frontal_face_detector()
        self.isloaded=True
        return True 
        
    def configure(self):
        return 0 
    
    def clean (self):
        '''
         clean the loaded model 
         
        '''
        self.predictor=None
        self.detector=None
        self.is_loaded =False
