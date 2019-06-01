from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def styler():
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(226,226,1)))
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Block 2
    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Block 3
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Block 4
    model.add(Conv2D(512,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Block 5
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1000,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

style = styler()
