#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Edgard Ortiz, Henrique Mualem Marti, Matheus Dib, Fabio de Miranda"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import auxiliar as aux
from ipywidgets import widgets, interact, interactive, FloatSlider, IntSlider

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

ins_bgr = cv2.imread('insper.jpeg')
ins_gray = cv2.cvtColor(ins_bgr, cv2.COLOR_BGR2GRAY)

def distancia_objeto(i):
    # f é a distância focal em pixels
    # i é o comprimento da imagem em pixels
    # o é o comprimento do objeto em centímetros
    # d = f*o/i
    if i == 0:
        d = 0
    else:
        d = (923*38)/(2.75*i)
    return d

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Convert the frame to rgb and hsv
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    blur = cv2.GaussianBlur(img_rgb,(5,5),0)
    #blur = img_rgb
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    
    # Mascara do magenta '#ff00ff'
    magenta = '#ff00ff'
    hsv_magenta1, hsv_magenta2 = aux.ranges(magenta)
    hsv_magenta1 = (120, 50, 50)
    hsv_magenta2 = (170, 255, 255)
    mask_magenta = cv2.inRange(img_hsv,hsv_magenta1,hsv_magenta2)
    selecao_magenta = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_magenta)
    segmentado_cor_magenta = cv2.morphologyEx(mask_magenta,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_magenta = cv2.bitwise_and(img_rgb, img_rgb, mask=segmentado_cor_magenta)
    
    # Mascara do ciano '#00ffff'
    ciano = '#00ffff'
    hsv_ciano1, hsv_ciano2 = aux.ranges(ciano)
    hsv_ciano1 = (60,50,50)
    hsv_ciano2 = (110,255,255)
    mask_ciano = cv2.inRange(img_hsv,hsv_ciano1,hsv_ciano2)
    selecao_ciano = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_ciano)
    segmentado_cor_ciano = cv2.morphologyEx(mask_ciano,cv2.MORPH_CLOSE,np.ones((10, 10)))
    selecao_ciano = cv2.bitwise_and(img_rgb, img_rgb, mask=segmentado_cor_ciano)
    

    
    # Mascara total magenta + ciano
    mask_total = mask_ciano + mask_magenta
    selecao_total = selecao_ciano + selecao_magenta


    circles = []
    

    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    

    
    
    
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)
    diam = 0
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        circ1 = circles[0][0]
        if len(circles[0])> 1:
            circ2 = circles[0][1]
            cv2.line(selecao_total,(circ1[0],circ1[1]),(circ2[0],circ2[1]),(255,0,0),5)
        
        diam = 0
        for i in circles[0,:]:
            print(i)
            
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
            if i[2] > diam:
                diam = i[2]
            

    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordas_color,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    dist = distancia_objeto(diam)
    print('dist={0:.2f} mm'.format(dist))
    
    # Tentando enxergar INSPER
    

    brisk = cv2.BRISK_create()
    
    MIN_MATCH_COUNT = 10
    cena_bgr = frame
    img_cena = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2GRAY)
    kp1, des1 = brisk.detectAndCompute(ins_gray ,None)
    kp2, des2 = brisk.detectAndCompute(img_cena,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    original_rgb = cv2.cvtColor(ins_bgr, cv2.COLOR_BGR2RGB)
    cena_rgb = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2RGB)
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    img3 = cv2.drawMatches(original_rgb,kp1,cena_rgb,kp2, good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    
    
    
    
    

    selecao_total = img3
    
    
    
    
    
    
    
    
    
    
    # Display the resulting frame
    cv2.imshow('Detector de circulos',selecao_total)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
