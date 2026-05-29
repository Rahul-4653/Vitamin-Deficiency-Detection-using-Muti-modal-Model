import os
import random
import numpy as np
import tensorflow as tf
import gradio as gr

from PIL import Image

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SKIN_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\rahul\Desktop\Vitamin B12\Model\skin_model.h5")
NAIL_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\rahul\Desktop\Vitamin B12\Model\nail_model.h5")
SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\rahul\Desktop\Vitamin B12\Model\symptom_model.h5")

IMG_SIZE = 224

# LOAD MODELS

def load_model_safe(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    return tf.keras.models.load_model(path, compile=False)


skin_model = load_model_safe(SKIN_MODEL_PATH)
nail_model = load_model_safe(NAIL_MODEL_PATH)
symptom_model = load_model_safe(SYMPTOM_MODEL_PATH)

# SYMPTOM SEVERITY ENCODING

severity_ranges = {

    "None": (0.00, 0.25),
    "Mild": (0.26, 0.50),
    "Moderate": (0.51, 0.75),
    "Severe": (0.76, 1.00)

}

def severity_to_numeric(level):

    low, high = severity_ranges[level]
    return random.uniform(low, high)

# DIET TYPE ENCODING

diet_map = {

    "Vegetarian": 0,
    "Non-Vegetarian": 1,
    "Vegan": 2

}

# IMAGE PREPROCESSING

def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# IMAGE VALIDATION

def image_type_warning(skin_img, nail_img):

    warning = ""

    try:

        skin_processed = preprocess_image(skin_img)
        nail_processed = preprocess_image(nail_img)

        skin_on_skin = np.max(skin_model.predict(skin_processed, verbose=0)[0])
        nail_on_skin = np.max(nail_model.predict(skin_processed, verbose=0)[0])

        skin_on_nail = np.max(skin_model.predict(nail_processed, verbose=0)[0])
        nail_on_nail = np.max(nail_model.predict(nail_processed, verbose=0)[0])

        # Soft warning only

        if nail_on_skin > (skin_on_skin + 0.20):
            warning += "\n⚠ Uploaded skin image may not be a proper skin image.\n"

        if skin_on_nail > (nail_on_nail + 0.20):
            warning += "\n⚠ Uploaded nail image may not be a proper nail image.\n"

    except:
        pass

    return warning

# B12 CLASSIFICATION

def classify_b12(score):

    if score >= 0.60:
        return "Deficient"

    elif score >= 0.40:
        return "Borderline"

    else:
        return "Normal"

# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_risk(score):

    if score >= 0.60:
        return "HIGH"

    elif score >= 0.35:
        return "MODERATE"

    else:
        return "LOW"

# =========================================================
# DIET RECOMMENDATION ENGINE
# =========================================================

def diet_recommendations(
    b12_risk,
    iron_risk,
    vitamin_d_risk,
    folate_risk,
    calcium_risk,
    protein_risk,
    malnutrition_risk
):

    recommendations = []

    if b12_risk >= 0.40:

        recommendations.append("""

Vitamin B12 Diet:
• Eggs
• Milk
• Yogurt
• Fish
• Chicken
• Fortified cereals
• Nutritional yeast

""")

    if iron_risk >= 0.35:

        recommendations.append("""

Iron / Anemia Diet:
• Spinach
• Lentils
• Jaggery
• Beans
• Red meat
• Beetroot

""")

    if vitamin_d_risk >= 0.35:

        recommendations.append("""

Vitamin D Diet:
• Morning sunlight
• Egg yolk
• Fortified milk
• Salmon
• Tuna

""")

    if folate_risk >= 0.35:

        recommendations.append("""

Folate Diet:
• Green leafy vegetables
• Citrus fruits
• Lentils
• Broccoli

""")

    if calcium_risk >= 0.35:

        recommendations.append("""

Calcium Diet:
• Milk
• Paneer
• Yogurt
• Sesame seeds
• Almonds

""")

    if protein_risk >= 0.35:

        recommendations.append("""

Protein Diet:
• Eggs
• Chicken
• Soybean
• Paneer
• Pulses

""")

    if malnutrition_risk >= 0.35:

        recommendations.append("""

General Nutrition:
• Balanced meals
• Adequate hydration
• Fruits and vegetables
• Regular protein intake

""")

    if len(recommendations) == 0:

        return "Maintain a healthy balanced diet."

    return "\n".join(recommendations)

# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_b12(

    skin_img,
    nail_img,

    fatigue,
    dizziness,
    numbness,
    pale_skin,
    memory_loss,
    weakness,
    headache,
    irritability,
    appetite_loss,
    diarrhea,
    vision_problem,
    tingling,
    breathing_problem,
    tongue_pain,
    weight_loss,

    age,
    diet_type,

    bone_pain,
    muscle_cramps,
    hair_fall,
    sunlight_exposure,
    muscle_weakness,
    diet_quality,
    recent_weight_change,
    sleep_quality

):

    try:

        if skin_img is None or nail_img is None:
            return "Please upload both skin and nail images."

        # =====================================================
        # IMAGE PROCESSING
        # =====================================================

        warning_message = image_type_warning(skin_img, nail_img)

        skin_input = preprocess_image(skin_img)
        nail_input = preprocess_image(nail_img)

        skin_pred = skin_model.predict(skin_input, verbose=0)[0]
        nail_pred = nail_model.predict(nail_input, verbose=0)[0]

        # =====================================================
        # ORIGINAL 17-FEATURE ANN INPUT
        # =====================================================

        symptom_vector = np.array([[

            severity_to_numeric(fatigue),
            severity_to_numeric(dizziness),
            severity_to_numeric(numbness),
            severity_to_numeric(pale_skin),
            severity_to_numeric(memory_loss),
            severity_to_numeric(weakness),
            severity_to_numeric(headache),
            severity_to_numeric(irritability),
            severity_to_numeric(appetite_loss),
            severity_to_numeric(diarrhea),
            severity_to_numeric(vision_problem),
            severity_to_numeric(tingling),
            severity_to_numeric(breathing_problem),
            severity_to_numeric(tongue_pain),
            severity_to_numeric(weight_loss),

            age,

            diet_map[diet_type]

        ]], dtype=np.float32)

        symptom_pred = symptom_model.predict(
            symptom_vector,
            verbose=0
        )[0]

        # =====================================================
        # NORMALIZE
        # =====================================================

        skin_pred = skin_pred / np.sum(skin_pred)
        nail_pred = nail_pred / np.sum(nail_pred)
        symptom_pred = symptom_pred / np.sum(symptom_pred)

        # =====================================================
        # MULTIMODAL FUSION
        # =====================================================

        final_pred = (

            0.4 * skin_pred +
            0.4 * nail_pred +
            0.2 * symptom_pred

        )

        # confidence smoothing

        final_pred = final_pred * 0.9

        # =====================================================
        # B12 PREDICTION
        # =====================================================

        deficiency_score = float(final_pred[1])

        b12_result = classify_b12(deficiency_score)

        # =====================================================
        # EXTRA HEALTH INPUTS
        # =====================================================

        bone_pain_score = severity_to_numeric(bone_pain)
        muscle_cramps_score = severity_to_numeric(muscle_cramps)
        hair_fall_score = severity_to_numeric(hair_fall)
        sunlight_score = severity_to_numeric(sunlight_exposure)
        muscle_weakness_score = severity_to_numeric(muscle_weakness)
        diet_quality_score = severity_to_numeric(diet_quality)
        weight_change_score = severity_to_numeric(recent_weight_change)
        sleep_quality_score = severity_to_numeric(sleep_quality)

        # =====================================================
        # SECONDARY DEFICIENCY RISK ENGINE
        # =====================================================

        fatigue_score = severity_to_numeric(fatigue)
        dizziness_score = severity_to_numeric(dizziness)
        pale_skin_score = severity_to_numeric(pale_skin)
        weakness_score = severity_to_numeric(weakness)
        tingling_score = severity_to_numeric(tingling)
        numbness_score = severity_to_numeric(numbness)

        # Iron / Anemia

        iron_risk = (

            0.30 * fatigue_score +
            0.25 * pale_skin_score +
            0.20 * dizziness_score +
            0.15 * weakness_score +
            0.10 * severity_to_numeric(breathing_problem)

        )

        # Vitamin D

        vitamin_d_risk = (

            0.35 * bone_pain_score +
            0.25 * muscle_weakness_score +
            0.20 * fatigue_score +
            0.20 * (1 - sunlight_score)

        )

        # Folate

        folate_risk = (

            0.30 * weakness_score +
            0.25 * appetite_loss.count("e")/10 +
            0.20 * fatigue_score +
            0.25 * weight_change_score

        )

        # Calcium

        calcium_risk = (

            0.40 * muscle_cramps_score +
            0.30 * bone_pain_score +
            0.30 * weakness_score

        )

        # Protein

        protein_risk = (

            0.35 * muscle_weakness_score +
            0.25 * hair_fall_score +
            0.20 * weakness_score +
            0.20 * diet_quality_score

        )

        # General Malnutrition

        malnutrition_risk = (

            0.25 * weight_change_score +
            0.25 * diet_quality_score +
            0.25 * fatigue_score +
            0.25 * sleep_quality_score

        )

        # =====================================================
        # CLASSIFY RISKS
        # =====================================================

        iron_label = classify_risk(iron_risk)
        vitamin_d_label = classify_risk(vitamin_d_risk)
        folate_label = classify_risk(folate_risk)
        calcium_label = classify_risk(calcium_risk)
        protein_label = classify_risk(protein_risk)
        malnutrition_label = classify_risk(malnutrition_risk)

        # =====================================================
        # DIETARY RECOMMENDATIONS
        # =====================================================

        diet_text = diet_recommendations(

            deficiency_score,
            iron_risk,
            vitamin_d_risk,
            folate_risk,
            calcium_risk,
            protein_risk,
            malnutrition_risk

        )

        # =====================================================
        # FINAL OUTPUT
        # =====================================================

        output = f"""


VITAMIN B12 ANALYSIS
==============================

{warning_message}

Primary Prediction:
{b12_result}



SECONDARY NUTRITIONAL ANALYSIS
==============================

Iron / Anemia Risk:
{iron_label} ({iron_risk*100:.2f}%)

Vitamin D Risk:
{vitamin_d_label} ({vitamin_d_risk*100:.2f}%)

Folate Deficiency Risk:
{folate_label} ({folate_risk*100:.2f}%)

Calcium Deficiency Risk:
{calcium_label} ({calcium_risk*100:.2f}%)

Protein Deficiency Risk:
{protein_label} ({protein_risk*100:.2f}%)

General Malnutrition Risk:
{malnutrition_label} ({malnutrition_risk*100:.2f}%)


DIETARY RECOMMENDATIONS
==============================

{diet_text}

"""

        return output

    except Exception as e:

        return f"Error: {str(e)}"

# =========================================================
# UI
# =========================================================

severity_levels = [

    "None",
    "Mild",
    "Moderate",
    "Severe"

]

interface = gr.Interface(

    fn=predict_b12,

    inputs=[

        gr.Image(type="pil", label="Upload Skin Image"),
        gr.Image(type="pil", label="Upload Nail Image"),

        gr.Dropdown(severity_levels, label="Fatigue"),
        gr.Dropdown(severity_levels, label="Dizziness"),
        gr.Dropdown(severity_levels, label="Numbness"),
        gr.Dropdown(severity_levels, label="Pale Skin"),
        gr.Dropdown(severity_levels, label="Memory Loss"),
        gr.Dropdown(severity_levels, label="Weakness"),
        gr.Dropdown(severity_levels, label="Headache"),
        gr.Dropdown(severity_levels, label="Irritability"),
        gr.Dropdown(severity_levels, label="Loss of Appetite"),
        gr.Dropdown(severity_levels, label="Diarrhea"),
        gr.Dropdown(severity_levels, label="Vision Problems"),
        gr.Dropdown(severity_levels, label="Tingling"),
        gr.Dropdown(severity_levels, label="Breathing Problems"),
        gr.Dropdown(severity_levels, label="Tongue Pain"),
        gr.Dropdown(severity_levels, label="Weight Loss"),

        gr.Slider(10, 80, label="Age"),

        gr.Dropdown(
            ["Vegetarian", "Non-Vegetarian", "Vegan"],
            label="Diet Type"
        ),

        gr.Dropdown(severity_levels, label="Bone Pain"),
        gr.Dropdown(severity_levels, label="Muscle Cramps"),
        gr.Dropdown(severity_levels, label="Hair Fall"),
        gr.Dropdown(severity_levels, label="Low Sunlight Exposure"),
        gr.Dropdown(severity_levels, label="Muscle Weakness Severity"),
        gr.Dropdown(severity_levels, label="Poor Diet Quality"),
        gr.Dropdown(severity_levels, label="Recent Weight Change"),
        gr.Dropdown(severity_levels, label="Poor Sleep Quality")

    ],

    outputs="text",

    title="B12Dx - Multimodal Nutritional Deficiency Detection System",

    description="""
Upload skin and nail images and select symptom severity levels.
The system predicts Vitamin B12 deficiency and provides
secondary nutritional deficiency risk analysis.
"""

)

interface.launch()
