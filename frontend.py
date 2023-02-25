import gradio as gr
import title_predict_api
import cv_api


def title_predict(Title):
    return title_predict_api.title_predict_view(Title)


def image_predict(Image):
    return cv_api.predict_image(Image)


text = gr.Interface(
    fn=title_predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter your title here"),
    outputs="text",
    allow_flagging=False
)

image = gr.Interface(
    fn=image_predict,
    inputs=gr.Image(),
    outputs="text",
    allow_flagging=False
)

demo = gr.TabbedInterface([text, image], ["Title2View", "Image2View"])
demo.launch(share=True)
