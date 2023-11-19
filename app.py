import gradio as gr
import torch
import os
from pyannote.audio import Pipeline
import time
import torchaudio


read_key = os.environ.get('HF_TOKEN', None)

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=read_key)

# pipeline.to(torch.device("cuda"))

def transcribe_speech(filepath):
    start_time = time.time()

    waveform, sample_rate = torchaudio.load(filepath)
    output_text = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Calculate processing time

    result = f"\nProcessing Time: {processing_time:.2f} seconds\n{output_text}"
    return result


    return output

app = gr.Blocks()

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources=["upload"], type="filepath"),
    outputs="text",
    live=True
)


with app:
    gr.TabbedInterface(
        [file_transcribe],
        ["Transcribe Audio File"],
    )

app.launch(debug=True, share=True)
