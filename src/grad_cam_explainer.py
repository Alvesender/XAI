import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import os
from PIL import Image



def make_gradcam_heatmap(img_array, model, pred_index=None):
    layer_names=[layer.name for layer in model.layers]
    last_conv_layer_name = layer_names[-6] #-6
    
    # First, we create a gradient model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    # Therefore we calculate the class activation heatmap.
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

    
def grad_cam(image, vehicle, file_name):
    grad_cam_path = r'dash_src/assets/images/grad_cam/' + vehicle + '/' + file_name
    if os.path.exists(grad_cam_path):
        return Image.open(grad_cam_path)
    # Prepare image
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    
    model = tf.keras.models.load_model('models/model.h5')
    
    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(image, model)




    alpha=0.4
    # Display heatmap
    # save_and_display_gradcam(image, heatmap)
    # Numpy to PIL image
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + image
    
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img[0])

    # Save the superimposed image
    superimposed_img.save(grad_cam_path)
    
    return Image.open(grad_cam_path)

    