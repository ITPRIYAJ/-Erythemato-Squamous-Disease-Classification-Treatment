import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import os
from fpdf import FPDF
import datetime

# Load the trained model
MODEL_PATH = r"D:\skin\new_disease_classification_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the file path.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define disease class labels
CLASS_NAMES = [
    "Psoriasis",
    "Seborrheic Dermatitis",
    "Lichen Planus",
    "Pityriasis Rosea",
    "Chronic Dermatitis",
    "Pityriasis Rubra Pilaris"
]

# Disease Information
DISEASE_INFO = {
    "Psoriasis": "A chronic autoimmune condition causing red, scaly patches on the skin.",
    "Seborrheic Dermatitis": "A common skin condition mainly affecting the scalp, causing scaly patches and dandruff.",
    "Lichen Planus": "An inflammatory skin condition that causes purplish, itchy, flat-topped bumps.",
    "Pityriasis Rosea": "A self-limiting skin rash characterized by scaly, pink patches on the trunk.",
    "Chronic Dermatitis": "A persistent skin inflammation causing itching, redness, and dryness.",
    "Pityriasis Rubra Pilaris": "A rare disorder leading to widespread scaling and redness."
}

# Treatment, home remedies, and precautions
TREATMENTS = {
    "Psoriasis": "Topical treatments such as corticosteroids, light therapy.",
    "Seborrheic Dermatitis": "Anti-fungal treatments, shampoos.",
    "Lichen Planus": "Steroid creams, oral medications.",
    "Pityriasis Rosea": "Self-limiting, no specific treatment required.",
    "Chronic Dermatitis": "Corticosteroids, moisturizers.",
    "Pityriasis Rubra Pilaris": "Vitamin A derivatives, topical steroids."
}

HOME_REMEDIES = {
    "Psoriasis": "Aloe vera gel, apple cider vinegar.",
    "Seborrheic Dermatitis": "Tea tree oil, coconut oil.",
    "Lichen Planus": "Turmeric paste, coconut oil.",
    "Pityriasis Rosea": "Oatmeal baths, avoid irritation.",
    "Chronic Dermatitis": "Aloe vera, coconut oil.",
    "Pityriasis Rubra Pilaris": "Olive oil, Vitamin E."
}

PRECAUTIONS = {
    "Psoriasis": "Avoid scratching, wear loose clothing.",
    "Seborrheic Dermatitis": "Avoid harsh shampoos, use mild soaps.",
    "Lichen Planus": "Avoid triggers such as stress, certain foods.",
    "Pityriasis Rosea": "Avoid hot showers, wear loose clothing.",
    "Chronic Dermatitis": "Keep skin moisturized, avoid allergens.",
    "Pityriasis Rubra Pilaris": "Avoid excessive sun exposure, moisturize regularly."
}

# Streamlit App Setup
st.title("🩺 Erythemato-Squamous Disease Classification & Treatment")
st.write("Upload a skin lesion image to classify its type and get treatment recommendations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("**Processing the image...**")

    try:
        # Load and process the image
        img = load_img(uploaded_file, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        predictions = model.predict(img_array)

        # Process the results
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # Display Disease Information
        st.success(f"**Predicted Disease:** {predicted_class} ✅")
        st.write(f"**Confidence Score:** {confidence:.2%}")
        st.subheader("📋 Disease Information")
        st.info(DISEASE_INFO[predicted_class])

        # Treatment Recommendation
        st.subheader("💊 Treatment Recommendations")
        st.success(TREATMENTS[predicted_class])

        # Home Remedies & Skincare Tips
        st.subheader("🌿 Home Remedies & Skincare Tips")
        st.info(HOME_REMEDIES[predicted_class])

        # Precautionary Measures
        st.subheader("⚠️ Precautionary Measures")
        st.warning(PRECAUTIONS[predicted_class])

        # PDF Report Generation with Timestamp in Filename
        def generate_pdf():
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Disease Classification Report", ln=True, align='C')
                pdf.ln(10)
                pdf.cell(200, 10, f"Disease: {predicted_class}", ln=True)
                pdf.multi_cell(0, 10, f"Description: {DISEASE_INFO[predicted_class]}")
                pdf.multi_cell(0, 10, f"Confidence Score: {confidence:.2%}")
                pdf.multi_cell(0, 10, f"Treatment: {TREATMENTS[predicted_class]}")
                pdf.multi_cell(0, 10, f"Precautions: {PRECAUTIONS[predicted_class]}")
                pdf.multi_cell(0, 10, f"Home Remedies: {HOME_REMEDIES[predicted_class]}")
                
                # Adding timestamp to the file name
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                pdf_filename = f"disease_report_{timestamp}.pdf"
                pdf.output(pdf_filename)
                return pdf_filename
            except Exception as e:
                st.error(f"An error occurred while generating the PDF: {e}")
                return None

        if st.button("📄 Download Report as PDF"):
            pdf_file = generate_pdf()
            if pdf_file:
                with open(pdf_file, "rb") as file:
                    st.download_button("Download PDF Report", file, file_name=pdf_file, mime="application/pdf")

    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")
