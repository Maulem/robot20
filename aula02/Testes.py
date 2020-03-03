
"""



import cv2
import time
# Cria o detector BRISK
brisk = cv2.BRISK_create()


# Configura o algoritmo de casamento de features que vÃª *como* o objeto que deve ser encontrado aparece na imagem
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Define o mÃ­nimo de pontos similares
MINIMO_SEMELHANCAS = 18


def find_good_matches(descriptor_image1, frame_gray):

    des1 = descriptor_image1
    kp2, des2 = brisk.detectAndCompute(frame_gray,None)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return kp2, good


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    original_rgb = cv2.imread("insper.jpeg")  # Imagem a procurar
    img_original = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2GRAY)
    #original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)


    # Encontra os pontos Ãºnicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            print("Problema para capturar o frame da camera")
            continue

        # Our operations on the frame come here
        frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2, good_matches = find_good_matches(des1, gray)

        if len(good_matches) > MINIMO_SEMELHANCAS:
            img3 = cv2.drawMatches(original_rgb,kp1,frame_rgb,kp2, good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Founded!',(0,50), font, 1,(0,0,255),2,cv2.LINE_AA)
            time.sleep(0.1)
        cv2.imshow("BRISK features", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
    
"""



# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:04:22 2020

@author: Admin
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Edgard Ortiz, Henrique Marti"


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



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Convert the frame to rgb and hsv
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     

    selecao_total = frame
    imgr = frame[:,:,0]
    plt.hist(imgr.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show
    # Display the resulting frame
    cv2.imshow('Mask', selecao_total)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
