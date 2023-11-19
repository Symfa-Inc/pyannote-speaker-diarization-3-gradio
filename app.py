import gradio as gr
import torch
import os
from pyannote.audio import Pipeline


read_key = os.environ.get('HF_TOKEN', None)

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=read_key)

pipeline.to(torch.device("cuda"))

def transcribe_speech(filepath):
    output = pipeline(filepath)
    return output

app = gr.Blocks()

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs="text",
)


with app:
    gr.TabbedInterface(
        [file_transcribe],
        ["Transcribe Audio File"],
    )

app.launch(debug=True)
gr.load("models/pyannote/speaker-diarization-3.1", hf_token=read_key).launch()