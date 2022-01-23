import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps



X=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")['labels']
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)


x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,train_size=7500,test_size=2500)
x_train_scale=x_train/255.0
x_test_scale=x_test/255.0

lr=LogisticRegression(solver='saga',multi_class='multinomial').fit(x_train_scale,y_train)
y_pred=lr.predict(x_test_scale)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

cam=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cam.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape

        upper_left= (int(width/2-100),int(height/2-100))
        bottom_right= (int(width/2+100),int(height/2+100))

        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        pil=Image.fromarray(roi)
        cvtimage=pil.convert('L')
        resizeimg=cvtimage.resize((28,28),Image.ANTIALIAS)
        invtimg=PIL.ImageOps.invert(resizeimg)
        pix=20
        
        lim=np.percentile(invtimg,pix)
        scaledValue=np.clip(invtimg-lim,0,255)
        max_pix=np.max(invtimg)
        scaledValue=np.asarray(scaledValue)/max_pix

        random1=np.array(scaledValue).reshape(1,784)
        ranpred=lr.predict(random1)
        print('The Predict Image is :- ',ranpred)
        
        cv2.imshow('Frame',gray)
        if cv2.waitKey(1000)& 0xFF==ord('q'):

            break




        
    except Exception as e:
        pass
        
cam.release()
cv2.destroyAllWindows()