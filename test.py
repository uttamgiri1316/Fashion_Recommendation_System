import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2


feature_list=pickle.load(open('embeddings.pkl','rb'))

filenames=pickle.load(open('filenames.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.traiable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img=image.load_img('sample/1163.jpg',target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
preprocess_img=preprocess_input(expanded_img_array)
result=model.predict(preprocess_img).flatten()
normalizend_result=result/norm(result)

neighbours=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbours.fit(feature_list)

distances,indices=neighbours.kneighbors([normalizend_result])

print(indices)
for file in indices[0][1:6]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)



#print(np.array(feature_list).shape)




















































