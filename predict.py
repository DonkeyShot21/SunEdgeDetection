from keras.models import model_from_json
import mahotas as mh
import numpy as np
import json

img_rows, img_cols = 28, 28

# Load architecture
with open('CNN_architecture.json', 'r') as architecture_file:
    model_architecture = json.load(architecture_file)

loaded_model = model_from_json(model_architecture)

# Load weights
loaded_model.load_weights('CNN_weights.h5')

# load image
#test_img = mh.colors.rgb2grey(mh.imread('enhanced.jpg'),dtype = np.float)
test_img = mh.imread('test2.jpg')
test_img = np.divide(test_img,255)

# Predict
result, patches = [], []
for x in range(img_rows//2, len(test_img)-img_rows//2, 10):
    patches = []
    for y in range(img_cols//2, len(test_img[0])-img_cols//2,10):
        patch = [test_img[x-img_rows//2:x+img_rows//2,y-img_cols//2:y+img_cols//2]]
        patches.append(patch)
    X_test = np.array(patches)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    result.append(np.uint8(loaded_model.predict_classes(X_test,len(X_test))*255))

result = np.array(result)
mh.imsave('result.jpg',result)
