import shap
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Create a SHAP (SHapley Additive exPlanations) explainer for model interpretation.
from PIL import Image


def shap_Explainer(x, class_names, vehicle, file_name):#, save_path):
    shap_path = r'dash_src/assets/images/shap/' + vehicle + '/' + file_name
    if os.path.exists(shap_path):
        return Image.open(shap_path)
    shap.initjs()
    model = tf.keras.models.load_model('models/model.h5')
    # Create a SHAP masker for image explanation using the "inpaint_telea" method.
    masker = shap.maskers.Image("inpaint_telea", x[0].shape)

    # Create a SHAP explainer to compute explanations for a machine learning model.
    explainer = shap.Explainer(model, masker, output_names = class_names)

    # Compute SHAP values for the input data using the SHAP explainer.
    shap_values = explainer(x)
    # Generate an image plot to visualize SHAP values.
    shap.image_plot(shap_values, show=False)
    plt.savefig(shap_path)
    # plt.savefig(save_path, format = 'png', bbox_inches='tight')
    #shap.summary_plot(shap_values)
    return Image.open(shap_path)