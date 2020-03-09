import os,cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageFile
import matplotlib.image as mpimg
ImageFile.LOAD_TRUNCATED_IMAGES=True 
rootDir ="D:/star"
img_rows=536
img_cols=640
batch_size = 4
num_classes = 7
epoch = 5
Learning_rate=0.001
def load_data(rootDir):
    data_list= []
    labels_list=[]
    for dir_,_, files in os.walk(rootDir):
        for fileName in files:
            relDir = os.path.relpath(dir_, rootDir)
            relFile = os.path.join(rootDir, relDir, fileName)
            data_list.append(relFile)
            np.random.shuffle(data_list)
        print(len(data_list))
    classes =['Blackberries','Blackcurrant','Blueberries','Cherry','Cranberry','Elderberry','Grape']
    print(classes)
    for imgname in data_list:
        imgname = imgname.replace("\n", "")
        for l in range(len(classes)):
            if classes[l] in imgname:
                label = classes.index(classes[l])
                labels_list.append(label)
                #print(label)
    return data_list, labels_list
X_train, y_train = load_data(rootDir+"/Train")
X_val, y_val = load_data(rootDir+"/Val")

def standardize(img):
    mean = np.mean(img, axis=(1, 2), keepdims=True) #Check the axis
    std = np.sqrt(((img - mean) ** 2).mean((1, 2), keepdims=True))
    img = ((img - mean)/std) * (1/255)
    return img

def generator(samples, label, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_label = label[offset:offset+batch_size]
            images = []
            label1 = []
            label1 = tf.keras.utils.to_categorical(batch_label, num_classes)
            for batch_sample in batch_samples:
                p = mpimg.imread(batch_sample)
                center_image=cv2.resize(p, (img_rows, img_cols))
                #label1 =(batch_label)
                images.append(center_image)
            X_try = np.array(images)
            X_try = standardize(X_try) #Normalize the data
            X_try = np.reshape(X_try, (-1, img_rows, img_cols, 1))
            y_try = np.array(label1) 
            #with open('C:/Users/EVIndia/Desktop/pe/' + 'filename.txt', 'a') as fo:
               # fo.write("%s\n " % (X_try))
                #fo.write("%s\n " % (y_try))
            #fo.close()
            yield X_try, y_try
# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_val, y_val, batch_size)