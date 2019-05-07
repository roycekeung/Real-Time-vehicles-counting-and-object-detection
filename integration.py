import cv2
import numpy as np
import vehicles
import time
import pandas as pd
import os, inspect
import sys


counts = pd.read_csv("store_counts.csv")
features_counts = counts.loc[:,:]                             
#3print (features_counts)
object_detection = pd.read_csv("store_cascade.csv")
features_object_detection = object_detection.loc[:,:]                           
#print (features_object_detection)
veh_time = features_counts.veh_time
veh_dir = features_counts.veh_dir
cascade_time = features_object_detection.cascade_time
cascade_veh = features_object_detection.cascade_veh


def multiplyList(myList, times_value) : 
      
    # Multiply elements one by one 
    result = []
    for x in myList: 
         y = times_value * x  
         result.append(int(y))
    return result  

cascade_time_list= list(cascade_time)
veh_type= []
veh_integrate_time= []
for n in veh_time:
    vehicle=[]
    for i in np.arange(abs(n-2), n+2, 0.01):
        s= round(i,2)*100
        a= multiplyList(cascade_time_list, 100)
        if s in a:
        
            #print (s)
            
            #print (cascade_time_list)
            #print (round(i,3))
            #print (cascade_veh[a.index(s)])
            vehicle.append(cascade_veh[a.index(s)])
        elif s not in a:
            pass

    if len(vehicle)==0:
        pass
    elif 'bus' in vehicle and 'motorbike' not in vehicle:
        veh_type.append('bus')
        veh_integrate_time.append(round((n+i)/2,2))
    elif all(element == 'private_car' for element in vehicle):
        veh_type.append('private_car')
        veh_integrate_time.append(round((n+i)/2,2))
    elif all(element == 'bus' for element in vehicle):
        veh_type.append('bus')
        veh_integrate_time.append(round((n+i)/2,2))
    elif all(element == 'motorbike' for element in vehicle):
        veh_type.append('motorbike')   
        veh_integrate_time.append(round((n+i)/2,2))     
    elif 'motorbike'in vehicle and 'bus' not in vehicle:
        veh_type.append('motorbike')
        veh_integrate_time.append(round((n+i)/2,2))
    elif 'motorbike'in vehicle and 'bus' in vehicle:
        veh_type.append('bus')
        veh_integrate_time.append(round((n+i)/2,2))
    else:
        pass


store_data={}
    
store_data['veh_integrate_time']=veh_integrate_time
store_data['veh_type']=veh_type
    
df_total= pd.DataFrame(store_data, columns= ['veh_integrate_time', 'veh_type'])
df_total= pd.concat([df_total, veh_dir[:(len(veh_type))]], axis=1)
df_total.to_csv('final_result.csv', index=False)




print ("Total number of vehicles that's counted by background_subtraction method:", len(veh_time), '\n')
print ('However, size of bus are too large and lighting intensity problems would influence the regconition of boundary formed in background subtraction method')
print ('Therefore, the error margin are applied.')
print ('Number of vehicles are usually counted with 1 or 2 more for each of buses passing through.')
print ('Thus, after readjust:')

print ('total number of vehicles:', len(veh_type))
print ('total number of private car:', veh_type.count('private_car'))
print ('total number of bus:', veh_type.count('bus'))
print ('total number of motorbike:', veh_type.count('motorbike'), '\n')

print (df_total, '\n')

