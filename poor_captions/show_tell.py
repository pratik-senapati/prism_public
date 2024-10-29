import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the InceptionV3 model
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Function to extract features from the image
def extract_features(image_path):
    image = preprocess_image(image_path)
    features = model.predict(image)
    return features

# Load your pre-trained captioning model
caption_model = load_model('path_to_your_caption_model.h5')

# Function to generate captions
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Function to process the Flickr8k test data and generate captions
def process_flickr8k_test_data(test_images_file, flickr_captions_file, output_file):
    # Load the test image filenames
    with open(test_images_file, 'r') as file:
        test_images = file.read().strip().split('\n')

    # Load the tokenizer
    with open('path_to_your_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    max_length = 34  # Define max length of the sequence

    # Generate captions for test images
    captions = {}
    for filename in test_images:
        image_path = os.path.join(flickr_images_dir, filename)
        photo = extract_features(image_path)
        caption = generate_caption(caption_model, tokenizer, photo, max_length)
        captions[filename] = caption

    # Save captions to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(captions, json_file, indent=4)

    print(f"Captions saved to {output_file}")

# Example usage
flickr_images_dir = "C:/Users/Pratik Senapati/Downloads/Flickr8k_Dataset/Flicker8k_Dataset"
flickr_captions_file = "C:/Users/Pratik Senapati/Downloads/Flickr8k_text/Flickr8k.token.txt"
test_images_file = "C:/Users/Pratik Senapati/Downloads/Flickr8k_text/Flickr_8k.testImages.txt"
output_file = "flickr8k_test_results.json"

# Process the dataset and generate captions
process_flickr8k_test_data(test_images_file, flickr_captions_file, output_file)
