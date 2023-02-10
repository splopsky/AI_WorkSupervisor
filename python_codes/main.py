import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
from sklearn.metrics import accuracy_score
from collections import deque
from tkinter import *
import tkinter.font
import time
import csv
import os

timestr = time.strftime("%Y%m%d_%H%M%S")
with open('data_{}.csv'.format(timestr), mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['waist', 'arms', 'legs', 'weight', 'owas_code'])

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open('waist2.pkl', 'rb') as f:
    model1 = pickle.load(f)
    
with open('legs2.pkl', 'rb') as f:
    model2 = pickle.load(f)
    
with open('basic_pose.pkl', 'rb') as f:
    model = pickle.load(f)
    
##허리 connection
waist_set = set(mp_holistic.POSE_CONNECTIONS)
waist_set.clear()
waist_set.update({(12,24),(11,23)})
    
##팔 connection
Larm_set = set(mp_holistic.POSE_CONNECTIONS)
Larm_set.clear()
Larm_set.update({(11,13),(13,15)})
Rarm_set = set(mp_holistic.POSE_CONNECTIONS)
Rarm_set.clear()
Rarm_set.update({(12,14),(14,16)})
arms_set= Larm_set.union(Rarm_set)

##다리 connection
Lleg_set = set(mp_holistic.POSE_CONNECTIONS)
Lleg_set.clear()
Lleg_set.update({(23,25),(25,27)})
Rleg_set = set(mp_holistic.POSE_CONNECTIONS)
Rleg_set.clear()
Rleg_set.update({(24,26),(26,28)})
legs_set= Lleg_set.union(Rleg_set)

    
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 



win = Tk()
win.geometry('400x200-650+300')
win.title("무게 입력")
font=tkinter.font.Font(family="맑은 고딕", size=20, slant="italic")

label = Label(win,text ="무게 입력: ",font=font)
label.place(x =80,y =60)
entry = Entry(win, width = 5,font=font)
entry.place(x =220,y =60)

def click(Return):
    global weight
    weight = int(entry.get())
    win.destroy()
    
button = Button(win,text="입력",command = click)
button.place(x=160,y=120)

win.bind('<Return>',click)
win.mainloop()

###################gui############

# Create pose container for action detection
pose_container = deque(maxlen=2)

stage1 = deque(['bending', 'arm_up2'])
stage2 = deque(['arm_up2', 'bending'])
stage1_count = 0
stage2_count = 0
pose_count = 0

#arm_up1가 인식되는 에러 커버 가능 code
stage1_1 = deque(['bending','arm_up1'])
stage1_2 = deque(['arm_up1','arm_up2'])

stage2_1 = deque(['arm_up2','arm_up1'])
stage2_2 = deque(['arm_up1','bending'])
stage1_1_count = 0
stage1_2_count = 0
stage2_1_count = 0
stage2_2_count = 0

total = 0

stages = [False, False, False]

# OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)

        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(255,191,0), thickness=2, circle_radius=2)
                                 )
      
        # Displaying the output
        try:
            # Extract Pose landmarks
            pose = results.pose_world_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            row = pose_row
            
            ##############################################3
            ##Calculate and "arm" angle##
            # Get right side coordinates
            right_knee = [pose[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x,pose[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [pose[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x,pose[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]
            right_hip = [pose[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,pose[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
            right_shoulder = [pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [pose[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,pose[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            
            # Calculate angle
            right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Get left side coordinates
            left_knee = [pose[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,pose[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [pose[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,pose[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
            left_hip = [pose[mp_holistic.PoseLandmark.LEFT_HIP.value].x,pose[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [pose[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,pose[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            
            # Calculate angle
            left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            ################################################### 
            ##  Make Predictions for OWAS##
            
            X = pd.DataFrame([row])
            
            # predict class for waist
            waist_class = model1.predict(X)[0]
            waist_prob = model1.predict_proba(X)[0]
            
            # predict class for legs
            legs_class = model2.predict(X)[0]
            legs_prob = model2.predict_proba(X)[0]
            
            
            ## Integrate body numbers ##
            # Waist classes
            if ((waist_class.split(' ')[0] == '3_1' or waist_class.split(' ')[0] == '3_2')):
                waist_display = '3'
            elif ((waist_class.split(' ')[0] == '4_1' or waist_class.split(' ')[0] == '4_2')):
                waist_display = '4'
            else : waist_display = waist_class.split(' ')[0]
                
            # Arm classes
            if (left_arm_angle < 60 and right_arm_angle < 60):
                arms_display = '1'
                arms_class = '1'
            elif (left_arm_angle >= 60 and right_arm_angle < 60):
                arms_display = '2'
                arms_class = '2_1'
            elif (left_arm_angle < 60 and right_arm_angle >= 60):
                arms_display = '2'
                arms_class = '2_2'
            else:
                arms_display = '3'
                arms_class = '3'
            
            # legs classes
            if ((legs_class.split(' ')[0] == '3_1' or legs_class.split(' ')[0] == '3_2')):
                legs_display = '3'
            elif ((legs_class.split(' ')[0] == '5_1' or legs_class.split(' ')[0] == '5_2')):
                legs_display = '5'
            elif (legs_class.split(' ')[0] == '6_1' or legs_class.split(' ')[0] == '6_2' or legs_class.split(' ')[0] == '6_3'):
                legs_display = '6'
            else : legs_display = legs_class.split(' ')[0]            
                 
            # weight classes
            if weight < 10:
                weight_display = '1'
            elif weight >= 20:
                weight_display = '3'
            else:
                weight_display = '2'            
                       
            ## Get the OWAS Code ##
            owas_code = waist_display + arms_display + legs_display + str(weight_display)
            
            # Put Results to csv file
             
            with open('data_{}.csv'.format(timestr), mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([waist_display, arms_display, legs_display, weight_display, owas_code])
           
                
            #######################################################
            ## Make Predictions for Action Detection ##
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            # Detect Actions with pose_container
            
            pose_container.append(body_language_class)
            
            # Picking up objects
            
            if pose_container == stage1:
                print('stage1 detected')
                stage1_count += 1
            if pose_container == stage2:
                print('stage2 detected')
                stage2_count += 1
            
            # arm_up1 error fixed code
            
            if pose_container == stage1_1:
                print('stage1_1 detected')
                stage1_count += 1
            if pose_container == stage1_2:
                print('stage1_2 detected')
                stage2_count += 1
            if pose_container == stage2_1:
                print('stage2_1 detected')
                stage1_count += 1
            if pose_container == stage2_2:
                print('stage2_2 detected')
                stage2_count += 1
            
            # Check once more
            if stage1_count > 0: 
                if stage1_count == stage2_count:
                    pose_count += 1
                    stage1_count = 0
                    stage2_count = 0
            ############################################
            
            ##danger!!
            
            #waist
            if waist_display == '3' :
                mp_drawing.draw_landmarks(image, results.pose_landmarks, waist_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                                 )  
            elif waist_display == '2' or waist_display == '4':
                mp_drawing.draw_landmarks(image, results.pose_landmarks, waist_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                 ) 
                
            #arms
            if arms_class == '2_1':
                mp_drawing.draw_landmarks(image, results.pose_landmarks, Larm_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                 )  
            elif arms_class == '2_2':
                mp_drawing.draw_landmarks(image, results.pose_landmarks, Rarm_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                 )               
            elif arms_class == '3':
                mp_drawing.draw_landmarks(image, results.pose_landmarks, arms_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                 )      
            #legs
            if (legs_display == '4' or legs_display == '5' or legs_display == '6'):
                if left_leg_angle < 170 and left_leg_angle >= 120:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, Lleg_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                                 )  
                elif left_leg_angle < 120:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, Lleg_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                 )  
                if right_leg_angle < 170 and right_leg_angle >= 120:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, Rleg_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                                 )  
                elif right_leg_angle < 120:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, Rleg_set , 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                                 )  
                            
            ##########################
            # Draw status box
            
            cv2.rectangle(image, (0,0), (640, 60),(245,117,16), -1)
            
            # Display Code
            
            cv2.putText(image, 'WAIST'
                        , (25,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, waist_display
                        , (40,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'ARMS'
                        , (90,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, arms_display
                        , (105,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'LEGS', 
                        (155,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, legs_display
                        , (170,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'OWAS CODE', 
                        (220,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, owas_code, 
                        (245,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Pose'
                        , (380,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (360,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'Action Count'
                        , (500,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(pose_count)+'times'
                        , (550,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
        
        
        cv2.imshow('Raw Webcam Feed', cv2.flip(image, 1))
        cv2.imshow('Raw Webcam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        


cap.release()
cv2.destroyAllWindows()