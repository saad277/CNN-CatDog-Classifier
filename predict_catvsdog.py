import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import set_printoptions
import cv2

#Initial Status

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3";


os.chdir(r"C:\Users\Ammad\Desktop\CNN- cat vs dog\datasets\catsvsdogs");





filepath="catsvsdogs_cnn_0.88.h5";

my_model=load_model(filepath);

print(my_model.summary())

def draw_test(name,prediction,input_image):
    BLACK=[0,0,0];
    
    

    expanded_image=cv2.copyMakeBorder(input_image,0,0,0,
                                      imageL.shape[0],
                                      cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image,str(prediction),(252,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,255,0),2)

    cv2.imshow(name,expanded_image)


for i in range(0,5):
    #rand=np.random.randint(0,len);
    path=r"C:\Users\Ammad\Desktop\CNN- cat vs dog\datasets\catsvsdogs\validation\cats\cat109.jpg"
    input_image=cv2.imread(path);

    

    
    imageL=cv2.resize(input_image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    input_image=input_image.reshape(1,150,150,3)

    print(input_image.shape)

    #prediction
    
    res=str(my_model.predict_classes(input_image,1,verbose=0)[0])

    print("result : ",res)

    res=str(res)
    
    if(res=="[1]"):
        res="dog";
    elif(res=="[0]"):
        res="cat"
    
    draw_test("prediction",res,imageL);

    cv2.waitKey(0)



cv2.destroyAllWindows();
    

    


    




    
