import streamlit as st
import requests

# Define the URL of the FastAPI endpoint
url = "http://localhost:8000/compare"

st.title("Image-Caption Similarity Checker")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Enter a reference caption
reference_caption = st.text_input("Enter the reference caption:")

if st.button("Compare"):
    if uploaded_file is not None and reference_caption:
        # Send the image and reference caption to the API
        files = {"file": uploaded_file.getvalue()}
        data = {"reference_caption": reference_caption}
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"Generated Caption: {result['reference_caption']}")
            st.write(f"Similarity Score: {result['similarity_score']:.4f}")
        else:
            st.write("Error: Could not get a response from the API.")
    else:
        st.write("Please upload an image and enter a reference caption.")
