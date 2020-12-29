import os
import argparse
import numpy as np
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

def keras2TFlite(model_path):
    #load a pre-trained model
    keras_model =tf.keras.models.load_model(model_path) #model_path is 'cifar10_resnet18_pruned.h5'

    #convert to tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    converter.target_spec.supported_types = [tf.float16] # float16 PTQ

    tflite_model = converter.convert()

    #save tflite model
    ext_idx=model_path.rfind('.')
    save_path=model_path[:ext_idx]+'_float16.tflite'
    with open(save_path, "wb") as f:
        f.write(tflite_model)



def load_dataset(dataset_name, batch_size):

    if dataset_name =='cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes=10
        img_shape=[32,32,3]

    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes=100
        img_shape=[32,32,3]
    elif dataset_name == 'imagenet':
        raise ValueError('Not yet implemented')
    else:
        raise ValueError('Invalid dataset name : {}'.format(dataset_name))
    
    img_shape=[32,32,3]
    normalize = [ [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    mean= normalize[0]
    std= normalize[1]

    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]


    y_train  = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train ,y_train , x_test, y_test,num_classes,img_shape 

def TFLiteInference(model_path,x_test,y_test):

    #Step 1. Load TFLite model and allocate tensors.
    interpreter=tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get indexes of input and output layers
    input_index= interpreter.get_input_details()[0]['index']
    output_index= interpreter.get_output_details()[0]['index']

    sum_correct=0.0
    sum_time=0.0
    for idx, data in enumerate(zip(x_test,y_test)):
        image=data[0]
        label=data[1]
        image=np.expand_dims(image, axis=0)
        
        s_time=time.time()
        #Step 2. Transform input data
        interpreter.set_tensor(input_index,image)
        #Step 3. Run inference
        interpreter.invoke()
        #Step 4. Interpret output
        pred=interpreter.get_tensor(output_index)
        
        sum_time+=time.time()-s_time
        if np.argmax(pred)== np.argmax(label):
            sum_correct+=1.0
    
    mean_acc=sum_correct / float(idx+1)
    mean_time=sum_time / float(idx+1)

    print(f'Accuracy of TFLite model: {mean_acc}')
    print(f'Inference time of TFLite model: {mean_time}')


def kerasInference(model_path,x_test,y_test):
    keras_model =tf.keras.models.load_model(model_path) 

    sum_correct=0.0
    sum_time=0.0
    for idx, data in enumerate(zip(x_test,y_test)):
        image=data[0]
        label=data[1]
        image=tf.expand_dims(image, axis=0)
        
        s_time=time.time()

        pred=keras_model(image)

        sum_time+=time.time()-s_time
        if np.argmax(pred)== np.argmax(label):
            sum_correct+=1.0

    mean_acc=sum_correct / float(idx+1)
    mean_time=sum_time / float(idx+1)

    print(f'Accuracy of keras model: {mean_acc}')
    print(f'Inference time of keras model: {mean_time}')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    keras_path='./cifar10_resnet18_pruned.h5'
    tflite_path='./cifar10_resnet18_pruned_float16.tflite'

    x_train ,y_train , x_test, y_test,num_classes,img_shape  = load_dataset('cifar10',1)
    
    keras2TFlite(keras_path)
    TFLiteInference(tflite_path,x_test,y_test)
    kerasInference(keras_path,x_test,y_test)

    




