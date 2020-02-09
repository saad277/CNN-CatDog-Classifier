import tensorflow as tf

import os
from os import environ , chdir
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import Input,models,layers,optimizers,callbacks
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Dropout,Activation,Flatten


environ["TFF_CPP_MIN_LOG_LEVEL"]="3";


chdir(r"C:\Users\Ammad\Desktop\CNN- cat vs dog\datasets\catsvsdogs");

#ImAGE data generators
train_datagen=ImageDataGenerator(rescale=1./255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)




train_generator=train_datagen.flow_from_directory(
    directory=r"C:\Users\Ammad\Desktop\CNN- cat vs dog\datasets\catsvsdogs\train",
    target_size=(150,150),
    batch_size=16,
    class_mode="binary")


valid_datagen=ImageDataGenerator(
    rescale=1./255,
    )



validation_generator=valid_datagen.flow_from_directory(
    
    directory=r"C:\Users\Ammad\Desktop\CNN- cat vs dog\datasets\catsvsdogs\validation",
    target_size=(150,150),
    batch_size=16,
    class_mode="binary",
    )

#model 
model=Sequential();

#1st layer
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

#2nd layer


model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


#3rd layer

model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))



#4th layer

model.add(Conv2D(filters=16,kernel_size=(2,2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#5th layer

model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))



#6th layer

model.add(Flatten())
model.add(Dense(units=64))
model.add(Activation("relu"))
model.add(Dropout(rate=0.4))
model.add(Dense(units=1))
model.add(Activation("sigmoid"))



#optimizer and metrices
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])


print(model.summary())


#Setting Callbacks for progress
check_p=callbacks.ModelCheckpoint(filepath="catsvsdogs_cnn_{val_accuracy:.2f}.h5",
                                  monitor="val_accuracy",verbose=1,
                                  save_best_only=True,save_weights_only=False);




reduce_lr=callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.81,
                                      patience=5,verbose=1,cooldown=2)


callback_list=[check_p,reduce_lr];


#Training Options

fit=model.fit_generator(generator=train_generator,
                           steps_per_epoch=20,
                           epochs=100,
                           verbose=1,
                           callbacks=callback_list,
                           validation_data=validation_generator,
                           validation_steps=3,)


#saving model


model.save(filepath=r"catsvsdogs_cnn.h5",overwrite=True)





print("Working....")
