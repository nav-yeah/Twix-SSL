import gradio as gr
from inference import predict

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(num_top_classes=10, label="Class Probabilities"),
        gr.Textbox(label="Result")
    ],
    title="Barlow Twins SSL — STL-10 Image Classifier",
    description="""
## Model Information

**Representation Learning:** Barlow Twins

**Encoder:** ResNet18

**Evaluation:** Linear Classifier / Linear Probe

**Number of Classes:** 10 (STL-10)
""",
)

if __name__ == "__main__":
    demo.launch()