import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml('mnist_784', version = 1, return_X_y = True)
X_train, X_test , y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
train_x_scale = X_train/255.0
test_x_scale = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class = 'multinomial').fit(train_x_scale,y_train)

def get_prediction(image):
    iM_PIL = Image.open(image)
    image_BW = iM_PIL.convert('L')
    image_BW_resized = image_BW.resize((20,28), Image.ANTIALIAS())
    pixel_filter = 20
    min_pixel = np.percentile(image_BW_resized, pixel_filter)
    image_inverted_scale = np.clip(image_BW_resized, min_pixel, 0, 255)
    max_pixel = np.max(image_BW_resized)
    image_inverted_scale = np.asarray(image_inverted_scale)/max_pixel
    test_sample = np.array(image_inverted_scale).reshape(1,784)
    test_predict = clf.predict(test_sample)
    return(test_predict[0])
