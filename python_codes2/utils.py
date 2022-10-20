import numpy as np
import csv
from pandas import *
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


data = read_csv("data_20221019_230529.csv")
 
# converting column data to list
waist_list = data['waist'].tolist()
arms_list = data['arms'].tolist()
legs_list = data['legs'].tolist()
weight_list = data['weight'].tolist()
owas_list = data['owas_code'].tolist()

arms_c = Counter(arms_list)
waist_c = Counter(waist_list)
legs_c = Counter(legs_list)
weight_c = Counter(weight_list)

arms_perc = []
waist_perc = []
legs_perc = []
weight_perc = []

waist = {
        1: 0,
        2: 0,
        3: 0,
        4: 0      
}

arms = {
        1: 0,
        2: 0,
        3: 0
}

legs = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }

weight = {   
        1: 0,
        2: 0,
        3: 0
}
            

            
def count_waist(container):
        for key in waist:
            waist[key] = container[key]

def count_arms(container):
        for key in arms:
            arms[key] = container[key]
            
def count_legs(container):
        for key in legs:
            legs[key] = container[key]
            
def count_weight(container):
        for key in weight:
            weight[key] = container[key]
            

# get statistics of each category
def get_statistics_waist():
    sum = 0
    for key in waist:
        sum += waist[key]  
    for key in waist:
        waist_perc.append(round((waist[key]/sum), 2))
        print(key, ':', round((waist[key]/sum * 100), 2), '%')
        

def get_statistics_arms():
    sum = 0
    for key in arms:
        sum += arms[key] 
    for key in arms:
        arms_perc.append(round((arms[key]/sum), 2))        
        print(key, ':', round((arms[key]/sum * 100), 2), '%')
    
        
        
def get_statistics_legs():
    sum = 0
    for key in legs:
        sum += legs[key]
    for key in legs:
        legs_perc.append(round((legs[key]/sum), 2))
        print(key, ':', round((legs[key]/sum * 100), 2), '%')
        
        
def get_statistics_weight():
    sum = 0
    for key in weight:
        sum += weight[key]
    for key in weight:
        weight_perc.append(round((weight[key]/sum), 2))
        print(key, ':', round((weight[key]/sum * 100), 2), '%')




        

count_arms(arms_c)
count_legs(legs_c)
count_waist(waist_c)
count_weight(weight_c)

print('Arms')
get_statistics_arms()
print('Legs')
get_statistics_legs()
print('Waist')
get_statistics_waist()
print('Weight')
get_statistics_weight()




# Plotting into graph - Arms

names = list(arms.keys())
values = arms_perc

plt.title('OWAS arm code statistics')
bars = plt.bar(range(len(arms)),values,tick_label=names)
plt.ylim(0, 1)

# plot 0-1 to percentage
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0%}'.format(x) for x in current_values])

#display percentage number
for bar in bars:
    width = bar.get_width()
    height = bar.get_height()
    x, y = bar.get_xy()
    plt.text(x+width/2,
             y+height*1.01,
             str(round((height*100),2))+'%',
             ha='center')
    
# save image and show
plt.savefig('amrs.png')
plt.show()

