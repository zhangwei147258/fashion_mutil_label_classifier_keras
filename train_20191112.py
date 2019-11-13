#! -*- coding:utf-8 

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer
from sklearn.model_selection import train_test_split
from cnn import SimpleNet
#from cnn import SmallerInceptionNet
from cnn import FashionNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from PIL import Image

# grab the image paths and randomly shuffle them
def load_data(data_dir, img_size):
    print("[INFO] loading images...")
    if not os.path.exists(data_dir):
        return None
    imagePaths = sorted(list(paths.list_images(data_dir)))
    random.seed(42)
    random.shuffle(imagePaths)

    datas = []
    labels = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
        if image is None:
           print(imagePath)
           continue
        # convert 8depth to 24 depth
        if len(image.shape)==2:
            with Image.open(imagePath) as img:
                rgb_img = img.convert('RGB')
                image = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)
        elif len(image.shape)==3: 
            if image.shape[2]==4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.shape[2]==1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = cv2.resize(image, img_size)
        image = img_to_array(image)
        datas.append(image)

        label = imagePath.split(os.path.sep)[-2].split("_")
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    datas = np.array(datas, dtype="float") / 255.0
    labels = np.array(labels)
    return datas, labels

def load_data_multilabels(data_dir, img_size):
    print("[INFO] loading images...")
    if not os.path.exists(data_dir):
        return None
    imagePaths = sorted(list(paths.list_images(data_dir)))
    random.seed(42)
    random.shuffle(imagePaths)

    datas = []
    category_labels = []
    color_labels = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        if image is None:
           print(imagePath)
           continue
        if image.shape[2]==4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.resize(image, img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        datas.append(image)

        (color_label, category_label) = imagePath.split(os.path.sep)[-2].split("_")
        category_labels.append(category_label)
        color_labels.append(color_label)

    # scale the raw pixel intensities to the range [0, 1]
    datas = np.array(datas, dtype="float") / 255.0
    category_labels = np.array(category_labels)
    color_labels = np.array(color_labels)
    return datas, category_labels, color_labels

# binarize the labels using scikit-learn's special multi-label
def binarize_multilabels_and_save(labels, path):
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    print(labels[:6])
    print('labels shape:', labels.shape)
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i + 1, label))
    with open(path, "wb") as f:
        f.write(pickle.dumps(mlb))
    return labels, len(mlb.classes_)
	
def binarize_labels_and_save(category_labels, color_labels, category_path, color_path):
    category_lb = LabelBinarizer()
    color_lb = LabelBinarizer()
    category_labels = category_lb.fit_transform(category_labels)
    color_labels = color_lb.fit_transform(color_labels)

    # loop over each of the possible class labels and show them
    for (i, label) in enumerate(category_lb.classes_):
        print("category {}. {}".format(i + 1, label))

    for (i, label) in enumerate(color_lb.classes_):
        print("color {}. {}".format(i + 1, label))

    with open(category_path, "wb") as f:
        f.write(pickle.dumps(category_lb))

    with open(color_path, "wb") as f:
        f.write(pickle.dumps(color_lb))
    return category_labels, color_labels, len(category_lb.classes_), len(color_lb.classes_)
	
# model_type='SimpleNet'  'SmallerInceptionNet'
def train_model(datas, labels, classes, finalAct='sigmoid', model_type='SimpleNet'):
    EPOCHS = 20
    INIT_LR = 1e-3
    BATCH_SIZE = 32
    INPUT_SHAPE = (96, 96, 3)
    (trainX, testX, trainY, testY) = train_test_split(datas, labels, test_size=0.2, random_state=42)
    if model_type == 'SimpleNet':
        simpleNet = SimpleNet(INPUT_SHAPE, classes, finalAct)
        model = simpleNet.build_model()
    else:
        smallerInceptionNet = SmallerInceptionNet()
        model = smallerInceptionNet.build_model(INPUT_SHAPE, classes, finalAct)

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(trainX, trainY, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, verbose=1,
						validation_data=(testX,testY))
                        
    model.save('trained_mode/' + '{}.h5'.format(model_type))

def train_fashionnet_model(datas, category_labels, color_labels, category_classes, color_classes, finalAct='softmaxt'):
    EPOCHS = 30
    INIT_LR = 1e-3
    BATCH_SIZE = 32
    INPUT_SHAPE = (96, 96, 3)
    (trainX, testX, trainCategoryY, testCategoryY, trainColorY, testColorY) = train_test_split(datas, category_labels, color_labels, test_size=0.2, random_state=42)

    fashionNet = FashionNet(INPUT_SHAPE, category_classes=category_classes, 
                               color_classes=color_classes, finalAct=finalAct)
    model = fashionNet.build_model()
    losses = { 'category_output':'categorical_crossentropy', 'color_output':'categorical_crossentropy' }
    loss_weights = {'category_output':1.0, 'color_output':1.0}

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt,loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

    history = model.fit(trainX, {'category_output': trainCategoryY, 'color_output':trainColorY},
						batch_size=BATCH_SIZE, epochs=EPOCHS,
						verbose=1,
                        validation_data=(testX, {'category_output': testCategoryY, 'color_output':testColorY}))

    model.save('trained_mode/' + '{}.h5'.format('FashionNet'))

    plot_fashionnet_loss_acc(history, EPOCHS)

def plot_loss_acc(history, EPOCHS):
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig('plot_loss_acc.png')

def plot_fashionnet_loss_acc(history, EPOCHS):
    loss_names = ['loss', 'category_output_loss', 'color_output_loss']
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

    for (i, l) in enumerate(loss_names):
        title = 'Loss for {}'.format(l) if l != 'loss' else 'Total loss'
        ax[i].set_title(title)
        ax[i].set_xlabel('Epoch #')
        ax[i].set_ylabel('Loss')
        ax[i].plot(np.arange(0, EPOCHS), history.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), history.history["val_"+l], label="val_"+l)
        ax[i].legend()
    plt.savefig('plot_fashionnet_losses.png')
    plt.close()
    '''
    accuray_names = ['category_output_acc', 'color_output_acc']
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
    for (i, l) in enumerate(accuray_names):
        title = 'Accuray for {}'.format(l)
        ax[i].set_title(title)
        ax[i].set_xlabel('Epoch #')
        ax[i].set_ylabel('Accuray')
        ax[i].plot(np.arange(0, EPOCHS), history.history[l], label=l)
        ax[i].plot(np.arange(0, EPOCHS), history.history["val_"+l], label="val_"+l)
        ax[i].legend()
    plt.savefig('plot_fashionnet_accs.png')
    plt.close()
    '''
	
def main():
    data_dir = './dataset'
    img_size = (96, 96)
    label_dir = './labels'
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    datas, labels = load_data(data_dir, img_size)
    labels, classes= binarize_multilabels_and_save(labels, os.path.join(label_dir, 'multi-label.pickle'))
    train_model(datas, labels, classes, finalAct='sigmoid', model_type='SimpleNet')

    '''
    datas, category_labels, color_labels = load_data_multilabels(data_dir, img_size)
    category_path = os.path.join(label_dir, 'category.pickle')
    color_path = os.path.join(label_dir, 'color.pickle')
    category_labels, color_labels, category_classes, color_classes = binarize_labels_and_save(category_labels, color_labels, category_path, color_path)
    train_fashionnet_model(datas, category_labels, color_labels, category_classes, color_classes, finalAct='softmax')
    '''

if __name__ == '__main__':
    main()
