import sys
import argparse
import time
import json

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

batch_size = 64
image_size = 224

class_names = {}

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    
    return image.numpy()

def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(expanded_image)
    probs, classes = tf.nn.top_k(prediction, k=top_k)
    
    probs = probs.numpy()
    classes = classes.numpy()
    
    return probs, classes
    
if __name__ == '__main__':
    
    print('predict.py is running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')
    
    args = parser.parse_args()
    print(args)
    print(f'arg1: {args.arg1}')
    print(f'arg2: {args.arg2}')
    print(f'top_k: {args.top_k}')
    print(f'category_names: {args.category_names}')
    
    image_path = args.arg1
    
    model = tf.keras.models.load_model(args.arg2, custom_objects={'KerasLayer':hub.KerasLayer})
    top_k = args.top_k
    
    if top_k is None:
        top_k = 5
    else:
        top_k = int(top_k)
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    probs, classes = predict(image_path, model, top_k)
    
    flower_classes = []
    for i in classes[0]:
        flower_classes.append(class_names[str(i+1)])
    
    print(probs)
    print(flower_classes)