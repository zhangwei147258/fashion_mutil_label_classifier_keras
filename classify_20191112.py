# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import tensorflow as tf

# load the image
# model_type = None, FashionNnet
def load_image(img_path, model_type=None):
    image = cv2.imread(img_path)
    output = imutils.resize(image, width=400)
    if model_type == 'FashionNnet':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image, output

def load_trained_model(img, model_path, labelbin_path):
    label_lb = pickle.loads(open(labelbin_path, "rb").read())
    model = load_model(model_path)
    proba = model.predict(img)[0]
    
    idxs = np.argsort(proba)[::-1][:2]
    label_1 = label_lb.classes_[idxs[0]]
    label_2 = label_lb.classes_[idxs[1]]
    
    proba_1 = proba[idxs[0]]
    proba_2 = proba[idxs[1]]
    
    result = (label_1, proba_1, label_2, proba_2)
    return result
    


# load the trained convolutional neural network 
def load_trained_fashionnet_model(img, model_path, categorybin_path, colorbin_path):
    category_lb = pickle.loads(open(categorybin_path, "rb").read())
    color_lb = pickle.loads(open(colorbin_path, "rb").read())
    
    model = load_model(model_path, custom_objects={'tf':tf})
    (category_proba, color_proba) = model.predict(img)

    category_idx = category_proba[0].argmax()
    color_idx = color_proba[0].argmax()
    category_label = category_lb.classes_[category_idx]
    color_label = color_lb.classes_[color_idx]
    
    category_proba = category_proba[0][category_idx]
    color_proba = color_proba[0][color_idx]
    result = (category_label, category_proba, color_label, color_proba)
    return result

def show_result(img, result):
    (label_1, proba_1, label_2, proba_2) = result
    text1 = "{}: {:.2f}%".format(label_1, proba_1*100)
    text2 = "{}: {:.2f}%".format(label_2, proba_2*100)
    
    cv2.putText(img, text1, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, text2, (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
def show_fashionnet_result(img, result):
    (category_label, category_proba, color_label, color_proba) = result
    category_text = "category: {}: {:.2f}%".format(category_label, category_proba*100)
    color_text = "color: {}: {:.2f}%".format(color_label, color_proba*100)

    cv2.putText(img, category_text, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, color_text, (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


if __name__=='__main__':
    test_dir = './examples'
    #model_type = 'FashionNnet'
    model_type = None
    for img in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img)
        if model_type == None:
            image,output = load_image(img_path)
            model_path = 'trained_mode/SimpleNet.h5'
            labelbin_path = './labels/multi-label.pickle'
            result = load_trained_model(image, model_path, labelbin_path)
            show_result(output, result)
        elif model_type == 'FashionNnet':
            image, output = load_image(img_path, model_type)
            model_path = 'trained_mode/FashionNet.h5'
            categorybin_path = './labels/category.pickle'
            colorbin_path = './labels/color.pickle'
            
            result = load_trained_fashionnet_model(image, model_path, categorybin_path, colorbin_path)
            show_fashionnet_result(output, result)

