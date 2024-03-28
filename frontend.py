import streamlit as st
import os
from PIL import Image
from backend import *
import pandas as pd
import threading, time
from queue import Queue

def save_uploaded_file(uploaded_file):
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def delete_image_file(file_path):
    os.remove(file_path)

def main():
    st.markdown("<h1 style='text-align: center;'>Food Image and Cuisine classifier</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("Welcome to the Food Image Classifier web application! Upload an image of food to get started.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            image_path = save_uploaded_file(uploaded_file)

            with st.spinner("Processing..."):
                results_queue = Queue()
                thread = threading.Thread(target=lambda: backend_thread(image_path, results_queue), daemon=True)
                thread.start()
                while thread.is_alive():
                    time.sleep(0.1)
                results = results_queue.get()
                st.success(f'Its a { " ".join(results[0][0].split("_")) } and its from the {results[1][0]} cuisine')
                st.write("Prediction results:")
                # col1, col2 = st.columns(2)
                # with col1:
                st.dataframe(pd.DataFrame(list(zip(results[0], results[2])), columns=["Top 5 Foods:", "Probabilities:"]), hide_index=True)
                # with col2:
                    # st.dataframe(pd.DataFrame(results[2], columns=[]), use_container_width=True, hide_index=True)
                # col1, col2 = st.columns(2)
                # with col1:
                st.dataframe(pd.DataFrame(list(zip(results[1], results[3])), columns=['Top 5 cuisines', "Probabilities:"]), hide_index=True)
                # with col2:
                    # st.dataframe(pd.DataFrame(results[3], columns=["Probabilities:"]), use_container_width=True, hide_index=True)
            
            delete_image_file(image_path)

    st.markdown("---")

def backend_thread(image_path, results_queue):
    food_model, cuisine_model = loadModel()
    results = predict_and_recommend(food_model, cuisine_model, image_path)
    results_queue.put(results)

if __name__ == "__main__":
    main()
