import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# print(cv2.__version__)

#图片读入
pic = cv2.imread("pic.jpg")
# print(pic)
# #改变大小
# picCopy = pic.copy()
# picCopy = cv2.resize(picCopy,(100,50))
# cv2.imshow('picCopy',picCopy)
# #截取
# picCut= pic[:100,:,:]#横
# picCut= pic[:,:140,:]#纵
# cv2.imshow('picCut',picCut)
#灰度
# cv2.imshow('picCut',cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY))
#图片翻转
# pic2 = cv2.flip(pic,-1)
# cv2.imshow('pic1',pic)
# cv2.imshow('pic2',pic2)
# cv2.waitKey()
# cv2.destroyAllWindows()

#划线
# picLine = cv2.line(pic,pt1=(50,10),pt2=(100,20),color=(0,0,0),lineType=cv2.LINE_8)
# picLine = cv2.arrowedLine(pic,pt1=(150,10),pt2=(100,20),color=(0,0,0))
# cv2.imshow('pl1',picLine)
# #画圆
# picCircle = cv2.circle(pic,(200,100),radius=55,color=(255,255,255))
# cv2.imshow('pc1',picCircle)
#t椭圆
# ellipse = cv2.ellipse(pic,(200,256),(100,50),30,0,360,255,3)
# cv2.imshow('ellipse',ellipse)
#画多边形
# points = np.random.rand(5,2)
# points = np.array(points*100,np.int32)
# print(points)
# # print(points[1:3,:1], points[1:3,1])
# # print(points[:,1],points[:,0])
# polyLine = cv2.polylines(pic,[points],True,(0, 0, 255),2,cv2.LINE_AA)
# cv2.imshow('polyLine',polyLine)
# #画五角星边形
# points = np.array([[120,150],[150,100],[180,150],[120,115],[180,115]])
# point5 = cv2.polylines(pic,[points],True,(255,0,0),2,cv2.LINE_AA)
# cv2.imshow('pic5',point5)
# #加内容
# picText = cv2.putText(pic,'ralph',(150,150),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
# cv2.imshow('picText',picText)

# cv2.waitKey()
# cv2.destroyAllWindows()

# 视频转图片
video = cv2.VideoCapture("C://Users//Administrator//Desktop//dog.mp4")
c = 1
if video.isOpened():
    ret,frame = video.read()
else:
    ret = False

print(video.get(3),video.get(4))#宽高
while ret:
    ret, frame = video.read()#是否打开  图片的每一帧
    # print(frame)
    if frame is None:
        ret = False
        continue
    cv2.imwrite('D://images/' + str(c) + '.jpg',frame)
    c = int(c) + 1

video.release()

# #图片转视频
# fps = 25
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# video_write = cv2.VideoWriter(filename="./result.avi",fourcc=fourcc,fps=fps,frameSize=(480,272))
# filePath = "images/"
# if os.path.exists(filePath):
#    files = os.listdir(filePath)
#    # print(files.__len__())
#    for i in range(files.__len__()):
#       pic = cv2.imread(filePath + str(i) + ".jpg")
#       cv2.circle(pic,(100,100),50,(24,45,41))
#       cv2.waitKey(100)
#       video_write.write(pic)
#       print(pic)


