import pickle
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from gpt4all import GPT4All
import os
# (The code at the top of the file remains exactly the same)
# ...
# --- 1. Define Categories and Features ---
CATEGORIES = {
    'art': ['art', 'ceramics', 'conceptual art', 'digital art', 'illustration', 'installations', 'mixed media', 'painting', 'performance art', 'public art', 'sculpture', 'social practice', 'textiles', 'video art'], 'comics': ['comics', 'anthologies', 'comic books', 'events', 'graphic novels', 'webcomics'], 'crafts': ['crafts', 'candles', 'crochet', 'diy', 'embroidery', 'glass', 'knitting', 'pottery', 'printing', 'quilts', 'stationery', 'taxidermy', 'weaving', 'woodworking'], 'dance': ['dance', 'performances', 'residencies', 'spaces', 'workshops'], 'design': ['design', 'architecture', 'civic design', 'graphic design', 'interactive design', 'product design', 'toys', 'typography'], 'fashion': ['fashion', 'accessories', 'apparel', 'childrenswear', 'couture', 'footwear', 'jewelry', 'pet fashion', 'ready-to-wear'], 'film & video': ['film & video', 'action', 'animation', 'comedy', 'documentary', 'drama', 'experimental', 'family', 'fantasy', 'festivals', 'horror', 'movie theaters', 'music videos', 'narrative film', 'romance', 'science fiction', 'shorts', 'television', 'thrillers', 'webseries'], 'food': ['food', 'bacon', 'community gardens', 'cookbooks', 'drinks', 'events', "farmer's markets", 'farms', 'food trucks', 'restaurants', 'small batch', 'spaces', 'vegan'], 'games': ['games', 'gaming hardware', 'live games', 'mobile games', 'playing cards', 'puzzles', 'tabletop games', 'video games'], 'journalism': ['journalism', 'audio', 'photo', 'print', 'video', 'web'], 'music': ['music', 'blues', 'chiptune', 'classical music', 'comedy', 'country & folk', 'electronic music', 'faith', 'hip-hop', 'indie rock', 'jazz', 'kids', 'latin', 'metal', 'pop', 'punk', 'r&b', 'rock', 'world music'], 'photography': ['photography', 'animals', 'fine art', 'nature', 'people', 'photobooks', 'places'], 'publishing': ['publishing', 'academic', 'anthologies', 'art books', 'calendars', "children's books", 'comedy', 'fiction', 'letterpress', 'literary journals', 'literary spaces', 'nonfiction', 'periodicals', 'poetry', 'radio & podcasts', 'translations', 'young adult', 'zines'], 'technology': ['gadgets', 'technology', '3d printing', 'apps', 'camera equipment', 'diy electronics', 'fabrication tools', 'flight', 'hardware', 'makerspaces', 'robots', 'software', 'sound', 'space exploration', 'wearables', 'web'], 'theater': ['theater', 'comedy', 'experimental', 'festivals', 'immersive', 'musical', 'plays', 'spaces']
}
RAW_FEATURES = [
    'country_AT', 'country_AU', 'country_BE', 'country_CA', 'country_CH', 'country_DE', 'country_DK', 'country_ES', 'country_FR', 'country_GB', 'country_GR', 'country_HK', 'country_IE', 'country_IT', 'country_JP', 'country_LU', 'country_MX', 'country_NL', 'country_NO', 'country_NZ', 'country_PL', 'country_SE', 'country_SG', 'country_SI', 'country_US', 'currency_AUD', 'currency_CAD', 'currency_CHF', 'currency_DKK', 'currency_EUR', 'currency_GBP', 'currency_HKD', 'currency_JPY', 'currency_MXN', 'currency_NOK', 'currency_NZD', 'currency_PLN', 'currency_SEK', 'currency_SGD', 'currency_USD', 'created_month_Apr', 'created_month_Aug', 'created_month_Dec', 'created_month_Feb', 'created_month_Jan', 'created_month_Jul', 'created_month_Jun', 'created_month_Mar', 'created_month_May', 'created_month_Nov', 'created_month_Oct', 'created_month_Sep'
]
COUNTRIES = sorted([f.split('_')[-1] for f in RAW_FEATURES if f.startswith('country_')])
CURRENCIES = sorted([f.split('_')[-1] for f in RAW_FEATURES if f.startswith('currency_')])
_months_unsorted = [f.split('_')[-1] for f in RAW_FEATURES if f.startswith('created_month_')]
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTHS = sorted(_months_unsorted, key=lambda m: month_order.index(m))

def get_expected_columns():
    cols = ['goal', 'duration', 'creator.backing_action_count']
    cols.extend([f'category.parent_name_{c.capitalize()}' for c in CATEGORIES.keys()])
    for parent, subcats in CATEGORIES.items():
        cols.extend([f'category.slug_{parent}/{s}' for s in subcats])
    cols.extend([f'created_month_{m}' for m in MONTHS])
    cols.extend([f'country_{c}' for c in COUNTRIES])
    cols.extend([f'currency_{c}' for c in CURRENCIES])
    return cols
EXPECTED_COLS = get_expected_columns()

def get_parent_categories(): return [key.capitalize() for key in CATEGORIES.keys()]
def get_sub_categories(parent_category): return CATEGORIES.get(parent_category.lower(), [])
def get_countries(): return COUNTRIES
def get_currencies(): return CURRENCIES
def get_months(): return MONTHS

def load_lgbm_model():
    model_path = 'models/LGBM_model.pkl'
    try:
        lgbm_model = joblib.load(model_path)
        print("LGBM model loaded successfully.")
        return lgbm_model
    except Exception as e:
        print(f"---!!! LGBM LOAD ERROR !!!---: {e}")
        return None

def load_llm_model():
    # Absolute path to the GGUF model file
    model_path = os.path.abspath("models/Llama-3.2-1B-Instruct-Q5_K_S.gguf")
    try:
        print("Loading Llama 3 GGUF model (gpt4all, CPU mode)...")
        llm = GPT4All(model_path)  # full path to your file
        print("Llama 3 model loaded successfully.")
        return llm
    except Exception as e:
        print(f"---!!! LLM LOAD ERROR !!!---: {e}")
        return None

def prepare_input_df(data):
    df1 = pd.DataFrame([data])
    df2 = pd.DataFrame(columns=EXPECTED_COLS)
    df2.loc[0] = 0
    for col in df1.columns:
        if col in df2.columns:
            df2[col] = df1[col].values
    return df2[EXPECTED_COLS]

def create_goal_plot(model, merged_df):
    goal_original = merged_df['goal'].values[0]
    num_points = 200
    spread = goal_original * 1.5
    goal_values = np.linspace(max(0, goal_original - spread), goal_original + spread, num_points)
    X_varied = pd.concat([merged_df] * num_points, ignore_index=True)
    X_varied['goal'] = goal_values
    probabilities = model.predict_proba(X_varied)[:, 1]
    smoothed_probabilities = gaussian_filter1d(probabilities, sigma=5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=goal_values, y=smoothed_probabilities, mode='lines', name='Success Probability'))
    fig.add_vline(x=goal_original, line_width=2, line_dash="dash", line_color="red", name="Your Goal")
    fig.update_layout(title_text='Probability vs. Goal Amount', xaxis_title='Goal Amount', yaxis_title='Probability of Success', legend_title="Legend")
    return fig

def create_month_plot(model, merged_df):
    month_cols = [f'created_month_{m}' for m in MONTHS]
    probabilities = []
    for month_col in month_cols:
        X_temp = merged_df.copy()
        for m_col in month_cols: X_temp[m_col] = 0
        X_temp[month_col] = 1
        prob = model.predict_proba(X_temp)[:, 1][0]
        probabilities.append(prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MONTHS, y=probabilities, mode='lines+markers', name='Success Probability'))
    fig.update_layout(title_text='Probability vs. Launch Month', xaxis_title='Month', yaxis_title='Probability of Success')
    return fig

def create_subcategory_plot(model, merged_df, parent_category, selected_sub_category):
    parent_key = parent_category.lower()
    sub_categories = get_sub_categories(parent_category)
    probabilities = []
    parent_col = f'category.parent_name_{parent_category}'
    for subcat in sub_categories:
        X_temp = merged_df.copy()
        for s_col in sub_categories:
            slug_col = f'category.slug_{parent_key}/{s_col}'
            if slug_col in X_temp.columns: X_temp[slug_col] = 0
        current_slug = f'category.slug_{parent_key}/{subcat}'
        if current_slug in X_temp.columns: X_temp[current_slug] = 1
        X_temp[parent_col] = 1
        prob = model.predict_proba(X_temp)[:, 1][0]
        probabilities.append(prob)
    selected_prob = probabilities[sub_categories.index(selected_sub_category)]
    colors = ['#1f77b4' if s == selected_sub_category else '#2ca02c' if p > selected_prob else '#d62728' for s, p in zip(sub_categories, probabilities)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub_categories, y=probabilities, marker_color=colors))
    fig.update_layout(title_text=f'Probability by Sub-Category for {parent_category}', xaxis_tickangle=-45, yaxis_title='Probability of Success')
    return fig

def prediction_pipeline(lgbm_model, parent_cat, sub_cat, country, currency, month, goal, prev_success_bool):
    """Main function to orchestrate prediction and graph generation."""
    if lgbm_model is None:
        # Return an HTML formatted error to match the output type
        return "<div style='color: red; text-align: center;'>LGBM Model not loaded. Please check logs.</div>", None, None, None

    if not all([parent_cat, sub_cat, country, currency, month, goal]):
        return "<div style='color: orange; text-align: center;'>Please fill in all prediction fields before running.</div>", None, None, None
        
    # --- Data Preparation (unchanged) ---
    parent_cat_key = parent_cat.lower()
    user_inputs = {
        "goal": goal,
        "duration": 30,
        "creator.backing_action_count": 5 if prev_success_bool else 0,
        f"category.parent_name_{parent_cat}": 1,
        f"category.slug_{parent_cat_key}/{sub_cat}": 1,
        f"created_month_{month}": 1,
        f"country_{country}": 1,
        f"currency_{currency}": 1,
    }

    # --- Prediction & Graphing (unchanged) ---
    merged_df = prepare_input_df(user_inputs)
    probability = lgbm_model.predict_proba(merged_df)[:, 1][0]
    
    goal_fig = create_goal_plot(lgbm_model, merged_df)
    month_fig = create_month_plot(lgbm_model, merged_df)
    subcat_fig = create_subcategory_plot(lgbm_model, merged_df, parent_cat, sub_cat)
    
    # --- NEW: HTML & CSS Styling for the Output ---
    
    # Determine color based on probability threshold (e.g., 50%)
    color = "green" if probability >= 0.5 else "red"
    prob_percent_str = f"{probability:.2%}"

    # Define CSS for a container that mimics a Gradio Label
    container_style = (
        "padding: 1.5rem; "
        "border: 1px solid var(--border-color-primary, #E5E7EB); " # Adapts to light/dark mode
        "border-radius: var(--radius-lg, 8px); "                   # Adapts to light/dark mode
        "text-align: center; "
        "background-color: var(--background-fill-secondary, #F9FAFB);" # Adapts to light/dark mode
    )

    # Define CSS for the main text and the colored percentage
    text_style = "font-size: 1.2rem;"
    percentage_style = f"color: {color}; font-weight: bold; font-size: 1.8rem;"

    # Construct the final HTML string
    probability_html = (
        f'<div style="{container_style}">'
        f'  <span style="{text_style}">The estimated probability of success is:</span><br>'
        f'  <span style="{percentage_style}">{prob_percent_str}</span>'
        '</div>'
    )

    return probability_html, goal_fig, month_fig, subcat_fig

def generate_blurb_pipeline(llm, parent_cat, sub_cat, user_prompt):
    if llm is None:
        return "LLM not loaded. Please check logs."
    if not all([parent_cat, sub_cat, user_prompt]):
        return "Please select a category, sub-category, and describe your company first."

    system_message = f"You are a helpful and friendly assistant who provides concise, catchy blurbs. The product category is {parent_cat} and the sub-category is {sub_cat}."
    full_prompt = f"{system_message}\nAdditional context from user: {user_prompt}"

    try:
        with llm.chat_session():
            response = llm.generate(full_prompt, max_tokens=512, temp=0.7)
        return response.strip()
    except Exception as e:
        return f"An error occurred during blurb generation: {e}"
