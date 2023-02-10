import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import cv2
import winsound as sd
import time
from sklearn.metrics import accuracy_score
from collections import deque
from utils import calculate_angle
import utils

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open('basic_pose.pkl', 'rb') as f:
    model = pickle.load(f)
    
 

cap = cv2.VideoCapture(0)
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

stages = [False, False, False]
glob_sum = 0
start = time.time()

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
            body_language_class = model.predict(X)[0] # the predicted pose
            body_language_prob = model.predict_proba(X)[0] # the predicted probability
            
            
            # code for pose count
            utils.count_pose(body_language_class)
            
            ##code for action detection##
            
            pose_container.append(body_language_class) # predicted pose into pose_container
            
            # 물체 옮기는 작업
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
                
            
            if stage1_count > 0: 
                if stage1_count == stage2_count:
                    pose_count += 1
                    stage1_count = 0
                    stage2_count = 0
                    
                    
            ### Code for Calculating Angle
            
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
            # cv2.putText(image, str(round(left_angle, 2)), 
            #                tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            
            
            # Get status box
            cv2.rectangle(image, (0,0), (300, 50), (245, 117, 16), -1)
            cv2.rectangle(image, (640,0), (540, 50), (245, 117, 16), -1)
            

            
            # Display Pose
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Stage
            cv2.putText(image, 'Action Count'
                        , (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(pose_count)
                        , (200,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            # Stop time
            end = time.time()
            total_time = end - start
            High_time = utils.return_max_time(total_time)
            
            ##경고 시스템 설정
            
            def beepsound():
                fr = 1000    # range : 37 ~ 32767
                du = 2000     # 1000 ms ==1second
                sd.Beep(fr, du) # winsound.Beep(frequency, duration)
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
            
            # Call Alert System
            color = (245, 117, 16)
            tcolor = (255, 255, 255)
            utils.sum_and_distribute(total_time,sum)
            
            # Display Alert System
            cv2.putText(image, utils.return_max_pose()
                        , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(utils.return_highest_time(total_time))+'sec'
                        , (510,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcolor, 2, cv2.LINE_AA)
            

        except:
            pass

                        
        cv2.imshow('Raw Webcam Feed', cv2.flip(image, 1))
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


## Utility Codes

print('total time: ', total_time)
print('return max pose: ', utils.return_max_pose())
total_sum = 0
total_sum = utils.sum_and_distribute(total_time)
print('highest_time: ', utils.return_highest_time(total_time))


# utils.sum_and_distribute(total_time, sum)