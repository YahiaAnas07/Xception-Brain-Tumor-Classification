import gradio as gr

model_path = '/content/xception.h5'
brain_model = load_model(model_path)

# Define the class names for the 4 classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_tumor(image):
    # Resize the image to the input shape required by the model
    image = Image.fromarray(image).resize((299, 299))
    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0
    # Ensure the image has 3 channels (RGB), if not convert it
    if image_array.shape[-1] != 3:
        image_array = np.stack((image_array,)*3, axis=-1)
    # Reshape the image to match the model's input shape
    image_array = image_array.reshape(-1, 299, 299, 3)
    # Predict the class using the model
    predictions = brain_model.predict(image_array)
    # Get the index of the highest prediction
    predicted_class_index = np.argmax(predictions)
    # Get the class name
    predicted_class = class_names[predicted_class_index]
    return f"Predicted Tumor Class: {predicted_class}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(),
    outputs=gr.Textbox(),
    title="Brain Tumor MRI Classification",
    description="Upload an MRI image to predict the brain tumor class."
)

# Launch the interface
iface.launch()