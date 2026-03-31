import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Loading model...")
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation='eager',
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16
)
model = model.eval().cuda()
print("Model loaded!")

import re

def clean_repetition(text):
    """Remove repetitive patterns from output"""
    lines = text.split('\n')
    seen = set()
    cleaned = []
    repeat_count = 0

    for line in lines:
        # Normalize line for comparison
        normalized = line.strip().lower()
        if normalized in seen:
            repeat_count += 1
            if repeat_count > 3:  # Allow max 3 repeats
                continue
        else:
            repeat_count = 0
            seen.add(normalized)
        cleaned.append(line)

    return '\n'.join(cleaned)

def ocr_image(image, prompt_type):
    if image is None:
        return "Please upload an image."

    # Save temp image
    temp_path = "/tmp/ocr_input.jpg"
    if isinstance(image, str):
        temp_path = image
    else:
        image.save(temp_path)

    # Select prompt based on type
    prompts = {
        "Document to Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "General OCR": "<image>\n<|grounding|>OCR this image.",
        "Free OCR (no layout)": "<image>\nFree OCR.",
        "Parse Figure/Chart": "<image>\nParse the figure.",
        "Describe Image": "<image>\nDescribe this image in detail."
    }
    prompt = prompts.get(prompt_type, prompts["Document to Markdown"])

    try:
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=temp_path,
            output_path='/tmp',
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            eval_mode=True
        )
        if result:
            # Clean up repetition
            result = clean_repetition(result)
            # Remove grounding tags if present
            result = re.sub(r'<\|ref\|>|<\|/ref\|>|<\|det\|>.*?<\|/det\|>', '', result)
        return result if result else "No text detected."
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"

# Create Gradio interface
with gr.Blocks(title="DeepSeek-OCR") as demo:
    gr.Markdown("# DeepSeek-OCR")
    gr.Markdown("Upload an image to extract text using DeepSeek-OCR")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_type = gr.Dropdown(
                choices=[
                    "Document to Markdown",
                    "General OCR",
                    "Free OCR (no layout)",
                    "Parse Figure/Chart",
                    "Describe Image"
                ],
                value="Document to Markdown",
                label="OCR Mode"
            )
            submit_btn = gr.Button("Run OCR", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="OCR Result", lines=20)

    submit_btn.click(fn=ocr_image, inputs=[image_input, prompt_type], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
