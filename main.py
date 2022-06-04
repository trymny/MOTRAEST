'''
==================================================
            Pose estimation of salmon
==================================================
 Info:

 Program by: Trym Nygård 
 Using: Erik Bochinski's IOU tracker
 Last updated: June 2022

 Multi object tracking method is based on the following paper: 
 http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf  

'''
# Import required libraries 
import matplotlib.pyplot as plt
import dataAugment as da
import pandas as pd
import numpy as np
import cv2 as cv
import iou 
import func as fc

from util import load_mot, save_to_csv

def dis2point(x,y,disp):
    
    vec_tmp = np.array([x,y,disp,1.0])
    vec_tmp = np.reshape(vec_tmp, (4,1))

    '''
    globQ = np.array([
                        np.array([1, 0, 0, -565.3385467529297]),
                       np.array([0, 1, 0, -410.1333084106445]),
                        np.array([0, 0, 0, 1289.887332633299]),
                        np.array([0, 0, 0.006655545399213341, -0.1802197918186371])
                    ])

    ''' 
    globQ = np.array([
                        np.array([1 ,0 ,0 , -640]),
                        np.array([0, 1 ,0, -409]),
                        np.array([0, 0, 0, 1372.484]),
                        np.array([0, 0, -1/150, 0])
                    ])
    
    vec_tmp = globQ@vec_tmp
    vec_tmp = np.reshape(vec_tmp, (1,4))[0]
    x = vec_tmp[0]
    y = vec_tmp[1]
    z = vec_tmp[2]
    w = vec_tmp[3]

    point = [x,y,z]/w

    return point # this is the 3D point

STEP = 3
NO_BLOCK = False
SUBPIX = True
dataFile = "forooshBalci"
FOLDER = "GT"

#********************STEP 1**********************
if STEP == 1:
    # Convert to correct data format
    df = da.prepData('data/'+FOLDER+'/LabelsLeft.csv','L', 3,1,250)
    df = da.prepData('data/'+FOLDER+'/LabelsRight.csv','R', 3,1,250)
#************************************************

#********************STEP 2**********************
if STEP == 2:
    # Import Data
    left = load_mot('data/'+FOLDER+'/filteredDataL.csv')
    right = load_mot('data/'+FOLDER+'/filteredDataR.csv')

    # Generate tracking data
    tracksL = iou.track_iou(left, 0.1, 0.4, 0.5, 5)
    tracksR = iou.track_iou(right, 0.1, 0.4, 0.5, 5)

    # Save to csv file
    save_to_csv('data/'+FOLDER+'/tracksL.csv',tracksL)
    save_to_csv('data/'+FOLDER+'/tracksR.csv',tracksR)
#************************************************

#********************STEP 3**********************
if STEP == 3:
    #Ground Truth
    colList = ['frame', 'id', 'x', 'y', 'w', 'h'] 
    leftData = pd.read_csv('data/'+FOLDER+'/filteredDataL.csv', usecols=colList)
    rightData = pd.read_csv('data/'+FOLDER+'/filteredDataR.csv', usecols=colList)

    frameNoL = leftData['frame']
    frameNoR = rightData['frame']
    xL = leftData['x']
    xR = rightData['x']
    yL = leftData['y']
    yR = rightData['y']
    id = leftData['id']

    if NO_BLOCK:
        from dataclasses import dataclass

        @dataclass
        class MatchedFeatures:
            frameNo: int
            xL: int
            yL: int
            xR: int
            yR: int

        # Matching
        Y_THRES = 1
        matchedList = []

        print(len(leftData), " ", len(rightData))
        for i in range(len(leftData)):
            for j in range(len(rightData)):
                if(frameNoL[i] == frameNoR[j] and yL[i] >= (yR[j]-Y_THRES) and yL[i] <= (yR[j]+Y_THRES)):
                    matchedList.append(MatchedFeatures(frameNoL[i],xL[i],yL[i],xR[j],yR[j]))

        print(len(matchedList))
     
    if NO_BLOCK:
        length = len(matchedList)
    else:
        length = len(xL)

    #Disparity 
    with open("data/"+FOLDER+"/trajectories/"+dataFile+".txt", "w") as c:
        for i in range(length):
   
            if NO_BLOCK:
                di  = int(matchedList[i].xL-matchedList[i].xR)
                point = dis2point(int(matchedList[i].xL),int(matchedList[i].yL),di )
                point = point/1000
                positionStr = str(frameNoL[i]) +' '+str(point[0]) +' '+ str(point[1]) +' '+ str(point[2]) + ' ' + str(id[i])
                if point[2] > 0 and point[2] < 2:
                    print(positionStr, file=c, flush=True)
            else:
                
                
                #Load images  -999 for 1000 for 1000to2000 NB: change file name
                imgL = cv.imread('images/stereo_left_'+FOLDER+'/L'+str(frameNoL[i])+'.jpg',0)
                imgR = cv.imread('images/stereo_right_'+FOLDER+'/R'+str(frameNoL[i])+'.jpg',0)
               

                #TM_CCOEFF_NORMED, TM_CCORR_NORMED, POC,  TM_SQDIFF_NORMED
                winL, winR, newXr, failed = fc.blockMatching(imgL,imgR,xL[i],yL[i],60, 60 , winFunc="False")     
             
                #Compute disparity with integer precition
                di  = int(xL[i]-newXr)
                            
                if SUBPIX:
                    subShiftX = fc.computeGradCorr(winL,winR,gradMethod="sobel")
                    point = dis2point(int(xL[i]),int(yL[i]),di -subShiftX)
                else:
                    point = dis2point(int(xL[i]),int(yL[i]),di )
               
                    
                if (failed == False):
                    point = point/1000
                    positionStr = str(frameNoL[i]) +' '+str(point[0]) +' '+ str(point[1]) +' '+ str(point[2]) + ' ' + str(id[i])
                    #if point[2] > 0 and point[2] < 2: #Remove for GT
                     
                    print(positionStr, file=c, flush=True)
#************************************************



