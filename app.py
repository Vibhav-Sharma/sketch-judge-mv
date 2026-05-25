import gradio as gr
import yaml
import json
import os
from pathlib import Path
import logging

# Try importing components from SketchJudge
from models import make_model
from data_utils import load_taxonomy

logging.basicConfig(level=logging.INFO)

# Load prompts
PROMPT_PATH = Path("prompts/prompt_baseline.yaml")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

# Try loading available taxonomies
TAXONOMIES = {
    "STEM (Default)": "data/SketchJudge_v1/taxonomy.json",
    "Biology": "data/SketchJudge_v1/biology_taxonomy.json"
}

def evaluate_sketch(api_key, model_name, taxonomy_choice, question_text, student_img, ref_img=None, question_img=None):
    if not api_key:
        return "Error: Please provide a Gemini API Key."
    if not student_img:
        return "Error: Please upload a student answer image."
    if not question_text:
        return "Error: Please provide a question text."

    # Load taxonomy string
    tax_path = TAXONOMIES.get(taxonomy_choice, TAXONOMIES["STEM (Default)"])
    try:
        taxonomy_dict = load_taxonomy(tax_path)
        taxonomy_text = taxonomy_dict.get('ALL', '')
    except Exception as e:
        return f"Error loading taxonomy: {e}"

    # Determine prompt key and build sample
    sample = {
        'query': question_text
    }
    
    has_ref = ref_img is not None
    has_q_img = question_img is not None

    if has_q_img:
        sample['image_0'] = question_img
        if has_ref:
            prompt_key = 'image_with_gt'
            sample['image_1'] = ref_img
            sample['image_2'] = student_img
        else:
            prompt_key = 'image_no_gt'
            sample['image_1'] = student_img
    else:
        if has_ref:
            prompt_key = 'no_image_with_gt'
            sample['image_0'] = ref_img
            sample['image_1'] = student_img
        else:
            prompt_key = 'no_image_no_gt'
            sample['image_0'] = student_img

    # Build prompt
    if prompt_key not in prompts:
        return f"Error: Prompt template {prompt_key} not found."
    
    query_template = prompts[prompt_key].replace("[[QUERY_EN]]", question_text)
    query_template = query_template.replace("[[TAXONOMY_BULLETED]]", taxonomy_text)
    sample['query'] = query_template

    # Initialize model
    try:
        model = make_model(
            "gemini",
            model_name=model_name,
            temperature=0.0,
            max_tokens=1024,
            api_key=api_key
        )
    except Exception as e:
        return f"Error initializing model: {e}"

    # Generate
    try:
        result_json_str = model.generate_from_sample(sample)
        # Attempt to pretty print if it's valid JSON
        start = result_json_str.find('{')
        end = result_json_str.rfind('}')
        if start != -1 and end != -1:
            clean_json_str = result_json_str[start:end+1]
            try:
                parsed = json.loads(clean_json_str)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except:
                pass
        return result_json_str
    except Exception as e:
        return f"Error during generation: {e}"

with gr.Blocks(title="SketchJudge Interactive Evaluation") as demo:
    gr.Markdown("# 🧑‍🏫 SketchJudge Interactive Evaluation")
    gr.Markdown("Upload a student's hand-drawn diagram and the question to evaluate it using AI.")

    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(label="Gemini API Key", type="password", placeholder="AIzaSy...")
            model_name = gr.Dropdown(choices=["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"], value="gemini-2.5-pro", allow_custom_value=True, label="Model")
            taxonomy = gr.Dropdown(choices=list(TAXONOMIES.keys()), value="Biology", label="Taxonomy")
            question = gr.Textbox(label="Question text", lines=3, placeholder="e.g. Draw a plant cell...")
            student_image = gr.Image(label="Student's Drawing (Required)", type="filepath")
            
            with gr.Accordion("Optional Images", open=False):
                question_image = gr.Image(label="Question Diagram (Optional)", type="filepath")
                reference_image = gr.Image(label="Reference Answer (Optional)", type="filepath")

            eval_btn = gr.Button("Evaluate Diagram", variant="primary")

        with gr.Column():
            output_json = gr.Code(label="Evaluation Result (JSON)", language="json")

    eval_btn.click(
        fn=evaluate_sketch,
        inputs=[api_key, model_name, taxonomy, question, student_image, reference_image, question_image],
        outputs=output_json
    )

if __name__ == "__main__":
    demo.launch(share=False)
