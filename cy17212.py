import cv2
import numpy as np
import os
import glob
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from sklearn import metrics

# read Haar-like feature classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')
#mouth_cascade= cv2.CascadeClassifier('mouth.xml')
nose_cascade= cv2.CascadeClassifier('nose.xml')

# read image files
files1 =glob.glob("C:\\Users\\enshu\\Desktop\\cy17248\\ekao\\*")
files2 =glob.glob("C:\\Users\\enshu\\Desktop\\cy17248\\bkao\\*")

feature = np.zeros((18, 2), dtype=np.float32)
label = np.zeros((18, 1), dtype=np.float32).T
count=0

def beauty_rate_measure(img):    
    # gray scale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x,y,w,h) in faces:
        # detected face enclose in a square
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # image(gray scale)
        roi_gray = gray[y:y+h, x:x+w]
        # image(colore scale)
        roi_color = img[y:y+h, x:x+w]
        # detect eyes in the face
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (ex,ey,ew,eh) in eyes:
            # detected eyes enclose in a square
            if y+ey+eh/2 < y+h/2:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # detect nose in the face
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (nx,ny,nw,nh) in nose:
            # detected nose enclose in a square
            if y+ny+nh/3 > y+h/3:
                 cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,0),2)
    return faces,eyes,nose

def create_golden_rate(faces,eyes,nose):
    if eyes[0,0] < eyes[1,0]:
        x1 = np.float32(faces[0,3])-(np.float32(nose[0,1])+np.float32(nose[0,3])*2/3)
        y1 = np.float32(faces[0,3])-x1-np.float32(eyes[0,1])
        z1 = np.float32(eyes[0,1])

        x1/=np.float32(faces[0,3])
        y1/=np.float32(faces[0,3])
        z1/=np.float32(faces[0,3])

        x2 = np.float32(eyes[1,2])
        y2 = np.float32(eyes[1,0]) - (np.float32(eyes[0,0])+np.float32(eyes[0,2]))
        z2 = np.float32(eyes[0,2])

        x2/=(np.float32(eyes[1,0])+np.float32(eyes[1,2]))-np.float32(eyes[0,0])
        y2/=(np.float32(eyes[1,0])+np.float32(eyes[1,2]))-np.float32(eyes[0,0])
        z2/=(np.float32(eyes[1,0])+np.float32(eyes[1,2]))-np.float32(eyes[0,0])
    else :
        x1 = np.float32(faces[0,3])-(np.float32(nose[0,1])+np.float32(nose[0,3])*2/3)
        y1 = np.float32(faces[0,3])-x1-np.float32(eyes[1,1])
        z1 = np.float32(eyes[0,1])

        x1/=np.float32(faces[0,3])
        y1/=np.float32(faces[0,3])
        z1/=np.float32(faces[0,3])

        x2 = np.float32(eyes[0,2])
        y2 = np.float32(eyes[0,0]) - (np.float32(eyes[1,0])+np.float32(eyes[1,2]))
        z2 = np.float32(eyes[1,2])

        x2/=(np.float32(eyes[0,0])+np.float32(eyes[0,2]))-np.float32(eyes[1,0])
        y2/=(np.float32(eyes[0,0])+np.float32(eyes[0,2]))-np.float32(eyes[1,0])
        z2/=(np.float32(eyes[0,0])+np.float32(eyes[0,2]))-np.float32(eyes[1,0])

    ave1=(x1+y1+z1)/3
    ave2=(x2+y2+z2)/3

    gr1=((x1-ave1)**2+(y1-ave1)**2+(z1-ave1)**2)/3
    gr2=((x2-ave2)**2+(y2-ave2)**2+(z2-ave2)**2)/3

    #print(x1, y1, z1, x2, y2, z2)
    return gr1,gr2

def create_datasets(files,flag):
    global count,feature,label
    for fname in files:
        img = cv2.imread(fname)
        faces,eyes,nose = beauty_rate_measure(img)
        gr1,gr2 = create_golden_rate(faces,eyes,nose)

        # create feature vector
        feature[count, 0] = gr1
        feature[count, 1] = gr2

        label[ 0, count] = flag

        count+=1

        # show image
        #cv2.imshow('img',img)
        # if any key pressed,destroy window
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

create_datasets(files1,1)
create_datasets(files2,0)


knn = cv2.ml.KNearest_create()
knn.train(feature,cv2.ml.ROW_SAMPLE,label)


cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    if ret == False:
        ret,img = cap.read()
    faces,eyes,nose = beauty_rate_measure(img)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if eyes.shape[0]==2 and nose.shape[0]==1:
            print("OK")
            break

gr1,gr2 = create_golden_rate(faces,eyes,nose)

# create feature vector
c_feature= np.zeros((1, 2), dtype=np.float32)
c_feature[0, 0] = gr1
c_feature[0, 1] = gr2

ret,results,neighbor,dist = knn.findNearest(c_feature,3)

# plot results
print(feature)
print(label)
print(c_feature)
print(results)

# create data for plot features data and plot it
feature1 = feature[label.ravel() == 1]
feature2 = feature[label.ravel() == 0]

plt.scatter(feature1[:,0],feature1[:,1],c='b',marker='s',s=10)
plt.scatter(feature2[:,0],feature2[:,1],c='r',marker='^',s=10)
plt.scatter(c_feature[0,0],c_feature[0,1],c='y',marker='o',s=10)
plt.xlabel('gr1')
plt.ylabel('gr2')
plt.show()

