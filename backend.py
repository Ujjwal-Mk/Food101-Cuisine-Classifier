import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow_datasets as tfds
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import streamlit as st

def loadModel():
    return tf.keras.models.load_model("model_84_9.h5"), tf.keras.models.load_model("cuisine_79_6.h5")

def predict_and_recommend(food_model, cuisine_model, imagePath, similarity=5):
    dataset_info = tfds.builder('food101').info
    food_class_names = dataset_info.features['label'].names
    
    cuisine_class_names = ['american','british','chinese','dutch','french','greece','indian','italian',\
                            'japanese','korean','mediterranean','mexican','spanish','thai']
    
    def load_and_preprocess_image(path, normalise=False):
        img = tf.keras.utils.load_img(path)
        img = tf.expand_dims(tf.cast(tf.image.resize(img, [512, 512]), dtype=tf.float32), axis=0)
        return img/255. if normalise else img

    similarity = min(similarity, len(cuisine_class_names))

    imageTensor = load_and_preprocess_image(path = imagePath)
    
    food_pred_probs = food_model.predict(imageTensor, verbose=0)[0]
    cuisine_pred_probs = cuisine_model.predict(imageTensor, verbose=0)[0]
    
    topFoods, topCuisines = [], []

    for i in range(similarity):
        idxFood = tf.argmax(food_pred_probs)
        idxCuisine = tf.argmax(cuisine_pred_probs)

        topFoods.append(food_class_names[idxFood])
        topCuisines.append(cuisine_class_names[idxCuisine])

        food_pred_probs[idxFood] = -1 * float('inf')
        cuisine_pred_probs[idxCuisine] = -1 * float('inf')

    return topFoods, topCuisines

# print(predict_and_recommend(food_model, cuisine_model, image_path, 5))