import numpy as np
import winsound as sd
import time

poses = {
        'stand': 0,
        'sit_chair': 0,
        'sit_floor': 0,
        'raise_hand': 0,
        'kneeling': 0,
        'squatting': 0,
        'bending': 0,
        'arm_up1': 0,
        'arm_up2': 0,
    }


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                         np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def count_pose(pose):
    for key in poses:
        if (key == pose):
            poses[key] += 1


# total sum of number of all pose inputs
def sum_and_distribute(total_time):
    sum = 0
    for key in poses:
        sum += poses[key]
        

    # print('total sum: ', sum)
    # print('total time (sit_floor): ', poses['sit_floor'] / sum * total_time)
        
    return sum


def beepsound():
    fr = 1000    # range : 37 ~ 32767
    du = 2000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def return_max_time(total_time):
    max_ratio = max(poses.values())/ sum
    max_time = round(total_time * max_ratio, 2)
    
        
    return max_time
        
        
        
def return_max_pose():
    return max(poses, key=poses.get)

def return_highest_time(total_time):
    return round(total_time * (max(poses.values())/ sum_and_distribute(total_time)), 2)