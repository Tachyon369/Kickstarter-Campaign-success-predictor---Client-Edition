import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib
import traceback

# --- Configuration ---
try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    ROOT_DIR = os.path.abspath('.')

MODELS_OUTPUT_DIR = os.path.join(ROOT_DIR, 'models_output')

# --- Hardcoded Model Features (Configuration Complete) ---

# 1. Numerical Features
NUMERICAL_FEATURES = [
    "creator.backing_action_count",
    "duration",
    "fx_rate",
    "goal",
    "usd_exchange_rate",
]

# 2. Categorical Features and their possible values.
CATEGORICAL_FEATURES = {
    "category.parent_name": ['Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video', 'Food', 'Games', 'Journalism', 'Music', 'Photography', 'Publishing', 'Technology', 'Theater'],
    "category.slug": ['art', 'art/ceramics', 'art/conceptual art', 'art/digital art', 'art/illustration', 'art/installations', 'art/mixed media', 'art/painting', 'art/performance art', 'art/public art', 'art/sculpture', 'art/social practice', 'art/textiles', 'art/video art', 'comics', 'comics/anthologies', 'comics/comic books', 'comics/events', 'comics/graphic novels', 'comics/webcomics', 'crafts', 'crafts/candles', 'crafts/crochet', 'crafts/diy', 'crafts/embroidery', 'crafts/glass', 'crafts/knitting', 'crafts/pottery', 'crafts/printing', 'crafts/quilts', 'crafts/stationery', 'crafts/taxidermy', 'crafts/weaving', 'crafts/woodworking', 'dance', 'dance/performances', 'dance/residencies', 'dance/spaces', 'dance/workshops', 'design', 'design/architecture', 'design/civic design', 'design/graphic design', 'design/interactive design', 'design/product design', 'design/toys', 'design/typography', 'fashion', 'fashion/accessories', 'fashion/apparel', 'fashion/childrenswear', 'fashion/couture', 'fashion/footwear', 'fashion/jewelry', 'fashion/pet fashion', 'fashion/ready-to-wear', 'film & video', 'film & video/action', 'film & video/animation', 'film & video/comedy', 'film & video/documentary', 'film & video/drama', 'film & video/experimental', 'film & video/family', 'film & video/fantasy', 'film & video/festivals', 'film & video/horror', 'film & video/movie theaters', 'film & video/music videos', 'film & video/narrative film', 'film & video/romance', 'film & video/science fiction', 'film & video/shorts', 'film & video/television', 'film & video/thrillers', 'film & video/webseries', 'food', 'food/bacon', 'food/community gardens', 'food/cookbooks', 'food/drinks', 'food/events', "food/farmer's markets", 'food/farms', 'food/food trucks', 'food/restaurants', 'food/small batch', 'food/spaces', 'food/vegan', 'games', 'games/gaming hardware', 'games/live games', 'games/mobile games', 'games/playing cards', 'games/puzzles', 'games/tabletop games', 'games/video games', 'journalism', 'journalism/audio', 'journalism/photo', 'journalism/print', 'journalism/video', 'journalism/web', 'music', 'music/blues', 'music/chiptune', 'music/classical music', 'music/comedy', 'music/country & folk', 'music/electronic music', 'music/faith', 'music/hip-hop', 'music/indie rock', 'music/jazz', 'music/kids', 'music/latin', 'music/metal', 'music/pop', 'music/punk', 'music/r&b', 'music/rock', 'music/world music', 'photography', 'photography/animals', 'photography/fine art', 'photography/nature', 'photography/people', 'photography/photobooks', 'photography/places', 'publishing', 'publishing/academic', 'publishing/anthologies', 'publishing/art books', 'publishing/calendars', "publishing/children's books", 'publishing/comedy', 'publishing/fiction', 'publishing/letterpress', 'publishing/literary journals', 'publishing/literary spaces', 'publishing/nonfiction', 'publishing/periodicals', 'publishing/poetry', 'publishing/radio & podcasts', 'publishing/translations', 'publishing/young adult', 'publishing/zines', 'technology', 'technology/3d printing', 'technology/apps', 'technology/camera equipment', 'technology/diy electronics', 'technology/fabrication tools', 'technology/flight', 'technology/gadgets', 'technology/hardware', 'technology/makerspaces', 'technology/robots', 'technology/software', 'technology/sound', 'technology/space exploration', 'technology/wearables', 'technology/web', 'theater', 'theater/comedy', 'theater/experimental', 'theater/festivals', 'theater/immersive', 'theater/musical', 'theater/plays', 'theater/spaces'],
    "country": ['AT', 'AU', 'BE', 'CA', 'CH', 'DE', 'DK', 'ES', 'FR', 'GB', 'GR', 'HK', 'IE', 'IT', 'JP', 'LU', 'MX', 'NL', 'NO', 'NZ', 'PL', 'SE', 'SG', 'SI', 'US'],
    "created_month": ['Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'],
    "currency": ['AUD', 'CAD', 'CHF', 'DKK', 'EUR', 'GBP', 'HKD', 'JPY', 'MXN', 'NOK', 'NZD', 'PLN', 'SEK', 'SGD', 'USD'],
}

# 3. All features in the exact order the model expects.
ALL_FEATURES_IN_ORDER = [
    "creator.backing_action_count",
    "duration",
    "fx_rate",
    "goal",
    "usd_exchange_rate",
    "category.parent_name",
    "category.slug",
    "country",
    "created_month",
    "currency",
]

# --- Helper Functions ---

def discover_saved_models():
    """Finds all compatible model files in the output directory."""
    if not os.path.exists(MODELS_OUTPUT_DIR):
        print(f"Warning: Directory not found, creating it: {MODELS_OUTPUT_DIR}")
        os.makedirs(MODELS_OUTPUT_DIR)
        return []
    model_files = [f.replace("_model.joblib", "") for f in os.listdir(MODELS_OUTPUT_DIR) if f.endswith("_model.joblib")]
    return sorted(model_files, reverse=True)


def run_prediction(model_base_name, *input_values):
    """
    Takes the model name and all input values, runs prediction, and returns the result with probability.
    """
    if not model_base_name:
        return "⚠️ Please select a model from the dropdown first.", ""
    if any(val is None or val == '' for val in input_values):
        return "⚠️ Please provide a value for all feature inputs.", ""

    model_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_base_name}_model.joblib")
    preprocessor_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_base_name}_preprocessor.joblib")
    target_encoder_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_base_name}_target_encoder.joblib")

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        target_encoder = joblib.load(target_encoder_path) if os.path.exists(target_encoder_path) else None

        input_data = pd.DataFrame([input_values], columns=ALL_FEATURES_IN_ORDER)
        processed_input = preprocessor.transform(input_data)

        # --- REVISED LOGIC: Get Prediction with Probability ---
        final_prediction_text = ""
        try:
            # Get the probability scores for each class (e.g., [0.1, 0.9])
            probabilities = model.predict_proba(processed_input)[0]
            
            # Get the predicted encoded label (e.g., 0 or 1)
            predicted_encoded_label = model.predict(processed_input)[0]
            
            # The encoded label *is* the index for the probabilities array.
            # We convert it to a standard Python int just to be safe.
            confidence_index = int(predicted_encoded_label)
            
            # Get the highest probability score (the confidence for the prediction)
            confidence = probabilities[confidence_index]
            
            # Get the friendly name of the predicted class (e.g., 'successful')
            # The target_encoder can transform the encoded label back to its string name.
            friendly_class_name = target_encoder.inverse_transform([predicted_encoded_label])[0] if target_encoder else predicted_encoded_label
            
            # Format the output string with the class name and its probability
            final_prediction_text = f"✅ Prediction: {friendly_class_name} ({confidence:.1%})"

        except AttributeError:
            # Fallback for models that don't have a .predict_proba() method
            prediction_raw = model.predict(processed_input)
            final_prediction = target_encoder.inverse_transform(prediction_raw)[0] if target_encoder else prediction_raw[0]
            final_prediction_text = f"✅ Prediction: {final_prediction} (Probability not available)"

        graph_placeholder = f"Confidence Score: {confidence:.2f}" if 'confidence' in locals() else "Graphing can be implemented here."

        return final_prediction_text, graph_placeholder

    except FileNotFoundError as e:
        error_msg = f"❌ Error: A required file was not found. Check names and paths.\nDetails: {e}"
        print(error_msg); traceback.print_exc()
        return error_msg, ""
    except Exception as e:
        error_msg = f"❌ An unexpected error occurred during prediction.\nDetails: {e}"
        print(error_msg); traceback.print_exc()
        return error_msg, ""


# --- Gradio UI Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="Model Predictor") as app:
    gr.Markdown("# 🤖 Kickstarter Project Predictor")
    gr.Markdown("Select a trained model and provide the project details to predict its success.")

    with gr.Row():
        saved_model_dd = gr.Dropdown(choices=discover_saved_models(), label="1. Select Saved Model", interactive=True)
        refresh_button = gr.Button("🔄 Refresh List")

    gr.Markdown("### 2. Provide Input Values")
    
    input_components = []
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Project Metrics")
            for feature in NUMERICAL_FEATURES:
                if feature == 'goal':
                    comp = gr.Number(label=feature, value=5000, interactive=True, info="Enter the funding goal.")
                elif feature == 'duration':
                     comp = gr.Slider(label=feature, minimum=1, maximum=90, value=30, step=1, interactive=True, info="Project duration in days.")
                else:
                    comp = gr.Number(label=feature, value=1.0, interactive=True)
                input_components.append(comp)

        with gr.Column(scale=2):
            gr.Markdown("#### Project Categories & Details")
            for feature, categories in CATEGORICAL_FEATURES.items():
                comp = gr.Dropdown(label=feature, choices=categories, interactive=True)
                input_components.append(comp)

    predict_button = gr.Button("3. Get Prediction", variant="primary")

    gr.Markdown("### 📈 Results")
    with gr.Row():
        prediction_output = gr.Label(label="Prediction Outcome & Confidence")
        graph_output = gr.Textbox(label="Log / Details", info="This area is for additional details or future visualizations.")

    # --- UI Interactions ---

    refresh_button.click(
        fn=lambda: gr.update(choices=discover_saved_models()),
        inputs=None,
        outputs=saved_model_dd,
        queue=False
    )
    
    predict_button.click(
        fn=run_prediction,
        inputs=[saved_model_dd] + input_components,
        outputs=[prediction_output, graph_output]
    )

if __name__ == "__main__":
    app.launch()