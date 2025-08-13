import gradio as gr
from model_utils import (
    get_parent_categories, get_sub_categories, get_countries, get_currencies, get_months,
    load_lgbm_model, prediction_pipeline,
    load_llm_model, generate_blurb_pipeline
)

# --- 1. Load Both Models at Startup ---
lgbm_model = load_lgbm_model()
llm = load_llm_model()

# --- 2. Define Initial UI State ---
parent_categories = get_parent_categories()
default_parent_category = parent_categories[0] if parent_categories else None
initial_sub_categories = get_sub_categories(default_parent_category) if default_parent_category else []
countries = get_countries()
currencies = get_currencies()
months = get_months()

# --- 3. Define UI Interaction Logic ---
def update_sub_category_choices(parent_category):
    sub_cats = get_sub_categories(parent_category)
    return gr.Dropdown(label="Sub-Category", choices=sub_cats, value=sub_cats[0] if sub_cats else None, interactive=True)
    
# --- Custom CSS for button styling ---
custom_css = """
#dark-mode-btn {
    max-width: 80px;
    flex-grow: 0;
    align-self: center;
}
"""

# --- 4. Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=custom_css) as app:
    
    # --- Title Bar with Dark Mode Toggle ---
    with gr.Row():
        gr.Markdown("# Kickstarter Campaign Success Predictor - Client Edition")
        dark_mode_btn = gr.Button("Toggle Theme ☀️/🌙", elem_id="dark-mode-btn")

    # --- Section for Predictive Analysis ---
    gr.Markdown("## Success Prediction Analysis")
    gr.Markdown("Fill in your project details to estimate its probability of success and see how different factors impact the outcome.")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Project Features", open=True):
                parent_category_input = gr.Dropdown(label="Parent Category", choices=parent_categories, value=default_parent_category)
                sub_category_input = gr.Dropdown(label="Sub-Category", choices=initial_sub_categories, value=initial_sub_categories[0] if initial_sub_categories else None)
                country_input = gr.Dropdown(label="Country", choices=countries, value=countries[0] if countries else None)
                currency_input = gr.Dropdown(label="Currency", choices=currencies, value=currencies[0] if currencies else None)
                month_input = gr.Dropdown(label="Launch Month", choices=months, value=months[0] if months else None)
                goal_input = gr.Slider(minimum=1000, maximum=250000, value=50000, step=1000, label="Goal", info="Select the funding amount required")
                previous_success_input = gr.Checkbox(label="I have had a successful Kickstarter before", value=False)
            predict_button = gr.Button("Analyze Project", variant="primary")
        with gr.Column(scale=2):
            probability_output = gr.HTML(label="Result")
            with gr.Tabs():
                with gr.TabItem("Goal vs. Probability"):
                    goal_plot_output = gr.Plot()
                with gr.TabItem("Month vs. Probability"):
                    month_plot_output = gr.Plot()
                with gr.TabItem("Sub-Category Comparison"):
                    subcategory_plot_output = gr.Plot()

    # --- Section for Generative AI Blurb ---
    gr.Markdown("<hr style='margin: 20px 0;'>")
    gr.Markdown("## Creative Blurb Generation")
    with gr.Row():
        with gr.Column(scale=1):
            blurb_prompt_input = gr.Textbox(label="What does your company Aim to do and what are your distinguishing characteristics?", lines=5, placeholder="e.g., We make high-quality, customizable mechanical keyboards for gamers...")
            blurb_button = gr.Button("Generate Blurb", variant="primary")
        with gr.Column(scale=2):
            blurb_output = gr.Textbox(label="Generated Blurb (Our Entry to \"Get out of Jail Free Card\")", interactive=False, lines=8)


    # --- 5. Wire Up Components ---
    parent_category_input.change(fn=update_sub_category_choices, inputs=parent_category_input, outputs=sub_category_input)
    
    predict_button.click(
        fn=prediction_pipeline,
        inputs=[gr.State(lgbm_model), parent_category_input, sub_category_input, country_input, currency_input, month_input, goal_input, previous_success_input],
        outputs=[probability_output, goal_plot_output, month_plot_output, subcategory_plot_output]
    )
    
    blurb_button.click(
        fn=lambda llm_in_list, p, sc, bp: generate_blurb_pipeline(llm_in_list[0], p, sc, bp),
        inputs=[gr.State([llm]), parent_category_input, sub_category_input, blurb_prompt_input],
        outputs=blurb_output
    )

    dark_mode_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            const dark = document.body.classList.toggle('dark');
            const btn = document.querySelector('#dark-mode-btn');
            if (btn) {
                btn.textContent = dark ? 'Change to Light Mode ☀️' : 'Change to Dark Mode 🌙';
            }
        }
        """
    )

# --- 6. Launch the Application ---
if __name__ == "__main__":
    # --- MODIFIED LINE ---
    # The 'inbrowser=True' argument will automatically open the UI in a new browser tab.
    app.launch(inbrowser=True)