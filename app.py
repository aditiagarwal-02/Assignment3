import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import io

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Load the trained model
model = keras.models.load_model("trained_model.h5")

# Define a function to get the feature maps for a given layer and input image
def get_feature_maps(model, layer_index, input_image):
    # Create a new model that outputs the feature maps for the given layer
    feature_extractor = keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    
    # Preprocess the input image and add a batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype("float32") / 255.0
    
    # Get the feature maps for the input image
    feature_maps = feature_extractor(input_image)
    
    return feature_maps

image_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

# Define the Streamlit app
def app():
    st.set_page_config(page_title="CNN Feature Map Visualizer", layout="wide")
    st.title("CNN Feature Map Visualizer")
    
    # Create a sidebar for the controls
    st.sidebar.title("Controls")
    
    # Create a dropdown menu to select the layer
    layer_names = [layer.name for layer in model.layers if 'flatten' not in layer.name and 'dense' not in layer.name and 'dense_1' not in layer.name]
    layer_name = st.sidebar.selectbox("Select a layer:", layer_names)
    layer_index = layer_names.index(layer_name)
    
    # Create a slider to select the image
    image_index = st.sidebar.slider("Select an image:", 0, len(x_test)-1, 0)
    image = x_test[image_index]
    placeholder = st.image(image, caption=f"Actual: {image_class[y_test[image_index][0]]}, Prediction: {image_class[np.argmax(model.predict(np.expand_dims(image, axis=0)))]}", width=200)
    
    # Add a button to show the model summary
    if st.sidebar.button("Show Model Summary"):
        with st.sidebar.expander("Model Summary", True):
            st.code(get_model_summary(model), language='python')
    
    if st.sidebar.button("Select a random image for me!"):
        placeholder.empty()
        image_index = np.random.randint(0, len(x_test))
        image = x_test[image_index]
        placebholder = st.image(image, caption=f"Actual: {image_class[y_test[image_index][0]]}, Prediction: {image_class[np.argmax(model.predict(np.expand_dims(image, axis=0)))]}", width=200)
    
    # Get the feature maps for the selected layer and image
    feature_maps = get_feature_maps(model, layer_index, image)
    
    num_channels = feature_maps.shape[-1]
    rows = int(np.ceil(num_channels/4))
    fig, axs = plt.subplots(nrows=rows, ncols=4, figsize=(8, 2*rows))
    for i in range(num_channels):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(feature_maps[0, :, :, i])
        axs[row, col].axis("off")
    fig.suptitle(f"Layer {layer_name}", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    app()
