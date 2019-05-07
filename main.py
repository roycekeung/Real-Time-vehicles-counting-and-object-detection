import cv2
import numpy as np
import vehicles
import time
import pandas as pd
import os, inspect
import sys

#cnt_up=0
cnt_down=0


cap=cv2.VideoCapture("VID_20190430_174950.mp4")


veh_time=[]
veh_dir= []
cascade_time=[]
cascade_veh=[]

#Get width and height of video

w=cap.get(3)    ####width for 720p = 1280
h=cap.get(4)    ####height for 720p = 720
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(2*(h/5))

line_down=int(3*(h/5))

up_limit=int(1*(h/5))
down_limit=int(4*(h/5))

print("Red line y:",str(line_down))
print("Blue line y:",str(line_up))
line_down_color=(255,0,0)          ### red  ###actually located upper
line_up_color=(0,0,255)          ### blue  ###actually located lower
pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
print (pts_L1)
print (pts_L1.shape)
pts_L1 = pts_L1.reshape((-1,1,2))
print (pts_L1)
print (pts_L1.shape)
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))


### adding haar cascade pretrained model
private_car_cascade= cv2.CascadeClassifier('cars.xml')
bus_cascade= cv2.CascadeClassifier('Bus_front.xml')
motorbike_cascade= cv2.CascadeClassifier('two_wheeler.xml')


#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)  ###Gray color region shows shadow region; detectShadows = True (which is so by default)

#Kernals    #### 用左dilation, 減少黑白frame裏面的noise（白點）
kernalOp = np.ones((3,3),np.uint8)    #### 好過下面 kernalOp2
kernalOp2 = np.ones((5,5),np.uint8)   #### 效果一般
kernalCl = np.ones((11,11),np.uint8)


font = cv2.FONT_HERSHEY_SIMPLEX    ### just font type
cars = []                          ### cars list 入邊係好多個 class car(i,xi,yi,max_age)
max_p_age = 5        ###????
pid = 1              ###????s


if not cap.isOpened():
    sys.exit('Camera did not provide frame.')


start_time= time.time()
while(cap.isOpened()):
    ret,frame=cap.read()              #### ret ussually return as True (coz your video is normally opened, frames r normally captured)
    frame = cv2.transpose(frame)          #### for rotating the video 90 degree
    frame = cv2.flip(frame,flipCode= 1)    #### flip repectively to y-axis
    #frame = cv2.flip(frame,flipCode= -1)   ### just like mirror reflect regarding to x, y axis


    for i in cars:
        i.age_one()                 #### def from vehicles.py (self-made)    cars.age+=1 ; return  True
    fgmask=fgbg.apply(frame)         #### 變做黑白
    fgmask2=fgbg.apply(frame)        #### 變做黑白

    if ret==True:

        #Binarization
        ##两个返回值，第一个retVal（得到的阈值值（在后面一个方法中会用到）），第二个就是阈值化后的图像
        ret,imBin=cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
        ret,imBin2=cv2.threshold(fgmask2,127,255,cv2.THRESH_BINARY)
        #OPening i.e First Erode the dilate   dilation, 減少黑白frame裏面object出面的noise（白點）
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)   ## in closing small holes inside the foreground objects, or small black points on the object.
        mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)  ##減少黑白frame裏面object出面的noise（白點）

        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.array(kernalCl))   ##減少黑白frame裏面object出面的noise（白點）
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)   ##減少黑白frame裏面object出面的noise（白點）

        ### haar cascade classification implementation
        gray= cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        private_car= private_car_cascade.detectMultiScale(gray, 1.2, 1)
        bus= bus_cascade.detectMultiScale(gray, 1.2, 1)
        motorbike= motorbike_cascade.detectMultiScale(gray, 1.1, 1)
        private_car_centre_y= False
        bus_centre_y= False
        motorbike_centre_y= False


        for (x,y,w,h) in private_car:
            font = cv2.FONT_HERSHEY_SIMPLEX
                                        #textposition   #textscale ##saddlebrown #thickness 
            cv2.putText(gray, 'private car', (x, y), font, 0.8, (19,69,139), 2, cv2.LINE_AA)
            cv2.rectangle(gray, (x,y), (x+w, y+h), (19,69,139), 2)
            cv2.circle(frame,(x+ int(w/2),y+ int(h/2)),5,(19,69,139),-1)
            private_car_centre_x, private_car_centre_y= [x+ int(w/2),y+ int(h/2)]

        for (x,y,w,h) in bus:
            font = cv2.FONT_HERSHEY_SIMPLEX                          ###pale blue 
            cv2.putText(gray, 'bus', (x, y), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(gray, (x,y), (x+w, y+h), (255, 255, 0), 2)
            cv2.circle(frame,(x+ int(w/2),y+ int(h/2)),5,(255, 255, 0),-1)
            bus_centre_x, bus_centre_y= [x+ int(w/2),y+ int(h/2)]

        for (x,y,w,h) in motorbike:
            font = cv2.FONT_HERSHEY_SIMPLEX                       ###dark blue
            cv2.putText(gray, 'motorbike', (x, y), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(gray, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame,(x+ int(w/2),y+ int(h/2)),5,(255, 0, 0),-1)
            motorbike_centre_x, motorbike_centre_y= [x+ int(w/2),y+ int(h/2)]



        #Find Contours
        ###a curve joining all the continuous points (along the boundary), having same color or intensity. 
        ##cv2.RETR_EXTERNAL表示只检测外轮廓   explain link=> ###https://blog.csdn.net/sunny2038/article/details/12889059
        ##cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            ### calculate the contour圈起身的 area
            area=cv2.contourArea(cnt)
            #print(area)
            if area>areaTH:
                ####Tracking######
                ### 中心點 calculate some features like center of mass of the object, area of the object etc.
                #### explain link https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])     ### centre point about x axis 個eqt 就係甘
                cy=int(m['m01']/m['m00'])     ### centre point about y axis 個eqt 就係甘
                ### (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
                x,y,w,h=cv2.boundingRect(cnt)

                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        ### self.x=xi  x is x_coordinate   ### self.y=yi  y is y_coordinate
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)    ####tracks.append([self.x, self.y])

                            #if i.going_UP(line_down,line_up)==True:
                                #cnt_up+=1                                   ###Locale’s appropriate date and time representation.  ##Mon Sep 30 07:06:05 2013
                                ##print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                                ##print (time.strftime("%H:%M:%S", time.localtime(time.time())))
                                ##print(time.time()-start_time)
                                #veh_time.append(round(time.time()-start_time, 2))
                                #veh_dir.append('up')
                            if i.going_DOWN(line_down,line_up)==True:
                                cnt_down+=1
                                #print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                                #print (time.strftime("%H:%M:%S", time.localtime(time.time())))
                                #print(time.time()-start_time)
                                veh_time.append(round(time.time()-start_time, 2))
                                veh_dir.append('down')
                            break
                        if i.getState()=='1':
                            if i.getDir()=='down'and i.getY()>down_limit:
                                i.setDone()              ### return True
                            elif i.getDir()=='up'and i.getY()<up_limit:
                                i.setDone()              ### return True
                        if i.timedOut():
                            index=cars.index(i)     ### return i 係第幾個 index number in list of cars[]
                            cars.pop(index)         ### return index,    cars 靜反 cars[裏面]除index element 外
                            del i                   ### delete i

                    if new==True: #If nothing is detected,create new
                        ###             centre of veh xi, yi ； max_p_age = 5
                        ###           Car(i,xi,yi,max_age)  input 緊value 入基本 init def
                        p = vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)              ### 擺落cars list 入邊係好多個 class car(i,xi,yi,max_age)
                        pid+=1
                ### cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])  ==》 Draws a circle
                ### 畫中心點出來
                ### Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn.
                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    ### (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        
        ###### 沒有用
        ##for i in cars:
            ##cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)


        if line_down+ 10>private_car_centre_y> line_down-50:
            if isinstance(private_car_centre_y, int)==isinstance(bus_centre_y, int):
                pass
            elif isinstance(private_car_centre_y, int)==isinstance(motorbike_centre_y, int):
                pass
            else:
                #print('This is a private car')
                #print (time.strftime("%H:%M:%S", time.localtime(time.time())))
                print('%.2f'% (time.time()-start_time))
                cascade_time.append(round(time.time()-start_time, 2))
                cascade_veh.append('private_car')

        elif line_down+ 10 >bus_centre_y> line_down-60:
            #print('This is a bus')
            #print (time.strftime("%H:%M:%S", time.localtime(time.time())))
            print('%.2f'% (time.time()-start_time))
            cascade_time.append(round(time.time()-start_time, 2))
            cascade_veh.append('bus')

        elif line_down+ 10 >motorbike_centre_y> line_down -80:
            #print('This is a motorbike')
            #print (time.strftime("%H:%M:%S", time.localtime(time.time())))
            print('%.2f'% (time.time()-start_time))
            cascade_time.append(round(time.time()-start_time, 2))
            cascade_veh.append('motorbike')
        else:
            pass

        #print (type(private_car))
        #print (private_car_centre_y, 'blue line:'+str(line_up), 'upper limit:'+str(up_limit))
        #print (private_car_centre_y, 'red line:'+str(line_down), 'down limit:'+str(down_limit))

        #str_up='UP: '+str(cnt_up)
        str_down='DOWN: '+str(cnt_down)
        frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        #frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        ####cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        ### org =》Bottom-left corner of the text string in the image. ； fontFace= font type
        #cv2.putText(frame, str_up, (10, 50), font, 2, (255, 255, 255), 6, cv2.LINE_AA)    ### make白底字先
        #cv2.putText(frame, str_up, (10, 50), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 100), font, 2, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 100), font, 2, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('Frame',frame)
        
        
        
        
        cv2.imshow('fg', fgmask)    ### display the banckground subtracting frames out
        cv2.imshow('gray conversion', gray)   ### display the gray conversion frame for the haar cascade
        ##cv2.imshow('gray conversion', mask)   ### display frame after dilation
        
        if cv2.waitKey(1)&0xff==ord('q'):
            break

    else:
        break

#### defining directory location
# script filename (usually with path)
script_name_and_path= inspect.getfile(inspect.currentframe())
# script directory
execution_directory= os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


print(veh_time)
print(cascade_time)
print(cascade_veh)

store_counts={}
store_cascade={}
store_counts['veh_time']=veh_time
store_counts['veh_dir']= veh_dir
store_cascade['cascade_time']=cascade_time
store_cascade['cascade_veh']=cascade_veh
df_counts= pd.DataFrame(store_counts, columns= ['veh_time', 'veh_dir'])
df_counts.to_csv('{}\\store_counts.csv'.format(execution_directory), index=False)
df_cascade= pd.DataFrame(store_cascade, columns= ['cascade_time','cascade_veh'])
df_cascade.to_csv('{}\\store_cascade.csv'.format(execution_directory), index=False)


def save_data():
    input_excel={}
    #input_excel['number of veh goes up']=np.array([cnt_up])
    input_excel['number of veh goes down']=np.array([cnt_down])
    df_total= pd.DataFrame(input_excel, columns= ['number of veh goes down'])
    df_total.to_csv('{}\\result.csv'.format(execution_directory), index=False)

save_data()


cap.release()
cv2.destroyAllWindows()









