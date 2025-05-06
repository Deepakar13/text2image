import gradio as gr
from generate import generate_image

def infer(prompt, scale, steps):
    return generate_image(prompt, guidance_scale=scale, steps=steps)

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Text to Image Generator using Stable Diffusion")
    with gr.Row():
        prompt = gr.Textbox(label="Enter Prompt", placeholder="A dragon flying over mountains at sunset")
    with gr.Row():
        scale = gr.Slider(minimum=1, maximum=15, value=7.5, label="Guidance Scale")
        steps = gr.Slider(minimum=10, maximum=100, value=50, label="Inference Steps")
    btn = gr.Button("Generate")
    output = gr.Image()

    btn.click(fn=infer, inputs=[prompt, scale, steps], outputs=output)

if __name__ == "__main__":
    demo.launch()