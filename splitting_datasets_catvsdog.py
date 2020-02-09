import os 
from os import listdir
from os.path import isfile,join
import cv2
import sys
import shutil
import numpy as np;

mypath="./datasets/catsvsdogs/images/";


file_names=[f for f in listdir(mypath) if isfile(join(mypath,f))]


print(str(len(file_names))+" images loaded");


## Extract 1000 for our training data and 500 for our validation data set 


dog_count=0;
cat_count=0;
training_size=1000;
test_size=500;
training_images=[];
training_labels=[];
test_images=[];
test_labels=[];
size=150;
interpolation=cv2.INTER_AREA;

dog_dir_train="./datasets/catsvsdogs/train/dogs/";
cat_dir_train="./datasets/catsvsdogs/train/cats/"
dog_dir_val="./datasets/catsvsdogs/validation/dogs/";
cat_dir_val="./datasets/catsvsdogs/validation/cats/";

def make_dir(directory):
    if(os.path.exists(directory)):
        shutil.rmtree(directory);
    os.makedirs(directory);

make_dir(dog_dir_train);
make_dir(cat_dir_train);
make_dir(cat_dir_val);
make_dir(dog_dir_val);


def getZeros(number):
    if(number>10 and number <100):
        return "0";
    if(number<10):
        return "00";
    else:
        return "";


for i,file in enumerate(file_names):
    

    if(file_names[i][0]=="d"):
        dog_count=dog_count+1;
        image=cv2.imread(mypath+file);
        
        
        image=cv2.resize(image,(150,150),interpolation)
        if(dog_count<=training_size):
            training_images.append(image);
            training_labels.append(1);
            zeros=getZeros(dog_count);
            cv2.imwrite(dog_dir_train + "dog" +str(zeros) +str(dog_count)+".jpg",image)

        if(dog_count>training_size and dog_count<=training_size+test_size):
            test_images.append(image);
            test_labels.append(1);
            zeros=getZeros(dog_count-1000);
            cv2.imwrite(dog_dir_val+"dog"+str(zeros)+str(dog_count-1000)+".jpg",image)


    if(file_names[i][0]=="c"):
        cat_count=cat_count+1;
        image=cv2.imread(mypath+file);
        
        image=cv2.resize(image,(150,150),interpolation)

        if(cat_count<=training_size):
            training_images.append(image);
            training_labels.append(0);
            zeros=getZeros(cat_count);
            cv2.imwrite(cat_dir_train + "cat"+str(zeros) + str(cat_count)+".jpg",image)

        if(cat_count>training_size and cat_count<=training_size+test_size):
            test_images.append(image);
            test_labels.append(0);
            zeros=getZeros(cat_count-1000);
            cv2.imwrite(cat_dir_val+"cat"+str(zeros)+str(cat_count-1000)+".jpg",image)



    if(dog_count==training_size+test_size and cat_count==training_size+test_size):
        break;


print("Training and Test Data Extraction Complete")



#Converting images to NPZ FILES

np.savez("cats_vs_dogs_training_data.npz",np.array(training_images))
np.savez("cats_vs_dogs_training_labels.npz",np.array(training_labels))
np.savez("cats_vs_dogs_test_data.npz",np.array(test_images))
np.savez("cats_vs_dogs_test_labels.npz",np.array(test_labels))


#Function for loading images

def load_data_training_and_test(datasetname):

    npzfile=np.load(datasetname+"_training_data.npz")
    train=npzfile["arr_0"];

    npzfile=np.load(datasetname+"_training_labels.npz");
    train_labels=npzfile["arr_0"];

    npzfile=np.load(datasetname+"_test_data.npz");
    test=npzfile["arr_0"];

    npzfile=np.load(datasetname+"_test_labels.npz");
    test_labels=npzfile["arr_0"];

    return (train,train_labels),(test,test_labels)




for i in range(1,11):
    random=np.random.randint(0,len(training_images))
    cv2.imshow("image"+str(i),training_images[random])
    if(training_labels[random]==0):
        print(str(i)+" - Cat");

    else:
        print(str(i)+" - Dog");
    cv2.waitKey(0);

cv2.destroyAllWindows();



    











