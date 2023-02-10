import pickle
import numpy as np
import pandas as pd
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
from sklearn.metrics import accuracy_score
from collections import deque
from utils import calculate_angle, beepsound
import winsound as sd
import time

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions


## open trained model
with open('basic_pose.pkl', 'rb') as f:
    model = pickle.load(f)
    
    
pose_container = deque(maxlen=2)

# stage1 = deque(['stand', 'bend_foward'])
# stage2 = deque(['bend_foward', 'crouch'])
# stage3 = deque(['crouch', 'stand'])
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

score = {'raise_hands': 0, 'bending': 0, 'squatting': 0, 'kneeling': 0}
total = 0

start = time.time()

stages = [False, False, False]


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
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        
        
        # Displaying the output
        try:
            # Extract Pose landmarks
            pose = results.pose_world_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            row = pose_row
                            
            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            # Print the predicted pose
            #print(body_language_class, body_language_prob)
            
            #############################
            # Code for Action Detection
            #############################
            
            pose_container.append(body_language_class)
            
            # Print pose container for action detection
            # print(pose_container)
            
            ## 물체 옮기는 작업
            if pose_container == stage1:
                print('stage1 detected')
                stage1_count += 1
            if pose_container == stage2:
                print('stage2 detected')
                stage2_count += 1
            
            #arm_up1 에러 커버가능 code
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
            
            
            if stage1_count > 0: #해결요망
                if stage1_count == stage2_count:
                    pose_count += 1
                    stage1_count = 0
                    stage2_count = 0
                    
                    
            ## 자세 비중 구하기
            if body_language_class == 'raise_hands':
                score['raise_hands'] += 1
            if body_language_class == 'bending':
                score['bending'] += 1                
            if body_language_class == 'squatting':
                score['squatting'] += 1                
            if body_language_class == 'kneeling':
                score['kneeling'] += 1 
            if body_language_class != 0:
                total += 1
            rate = max(score.values())/ total
            
            ##최대 자세지속시간 구하기
            end = time.time()
            realtime = end-start
            High_time = round(realtime * rate,2)
            
            
            ##경고 시스템 설정
            if High_time < 5:
                color = (245,117,16)
                tcolor = (255, 255, 255)                
            elif High_time >= 5 and High_time < 10:
                color = (0,230,247)
                tcolor = (70, 70, 70)                
            elif High_time >= 10 and High_time < 15:    
                color = (0,0,255)
                tcolor = (255, 255, 255)
            else:
                beepsound()
                break
                
            
            
            #Calculate and Display angle
            # Get right side coordinates
            right_shoulder = [pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [pose[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,pose[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [pose[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x,pose[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle
            right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Visualize angle
            cv2.putText(image, str(round(right_angle, 2)), 
                           tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA           
                       )

                        
            # Get left side coordinates
            left_shoulder = [pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [pose[mp_holistic.PoseLandmark.LEFT_HIP.value].x,pose[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [pose[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,pose[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            
            # Visualize angle
            cv2.putText(image, str(round(left_angle, 2)), 
                           tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            
            
            
            # Get status box
            cv2.rectangle(image, (0,0), (320, 60), (245, 117, 16), -1)
            cv2.rectangle(image, (490,0), (640, 100), color, -1)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'CLASS'
                        , (105,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)    
            
            #Display Stage
            cv2.putText(image, 'Pose Count'
                        , (200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(pose_count)
                        , (240,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, max(score, key=score.get)
                        , (510,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),1, cv2.LINE_AA)
            cv2.putText(image, str(High_time)+'sec'
                        , (510,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcolor, 2, cv2.LINE_AA)
        except:
            pass
        
        
        #cv2.imshow('Raw Webcam Feed', cv2.flip(image, -1))
        cv2.imshow('Raw Webcam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        


cap.release()
cv2.destroyAllWindows()