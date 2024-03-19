import streamlit as st
import os
from PIL import Image
from backend import *
import pandas as pd
import threading, time

# Function to save uploaded file to a temporary directory
def save_uploaded_file(uploaded_file):
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to delete the saved image file
def delete_image_file(file_path):
    os.remove(file_path)

def main():
    st.title("Food Image and Cuisine classifier")
    st.markdown("---")

    st.write("Welcome to the Food Image Classifier web application! Upload an image of food to get started.")

    # Add a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            image_path = save_uploaded_file(uploaded_file)

            with st.spinner("Processing..."):
                thread = threading.Thread(target=lambda: backend_thread(image_path), daemon=True)
                thread.start()
                while thread.is_alive():
                    time.sleep(0.1)
                st.success("Prediction completed!")
            
            delete_image_file(image_path)

    st.markdown("---")

def backend_thread(image_path):
    food_model, cuisine_model = loadModel()
    results = predict_and_recommend(food_model, cuisine_model, image_path)
    results = list(zip(*results))
    st.write("Prediction results:")
    st.dataframe(pd.DataFrame(results, columns=["Top 5 Foods:","Top 5 Cuisines:"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
