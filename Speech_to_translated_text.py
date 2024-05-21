import torchaudio
import torch
import gradio as gr
from transformers import SeamlessM4Tv2Model, AutoProcessor, SeamlessM4Tv2ForSpeechToText

import torchaudio, torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

#MODEL_All = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(DEVICE)

MODEL_TYPE = "S2TT"
if MODEL_TYPE == "S2TT":
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").to(DEVICE)
else:
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(DEVICE)
    

def reverse_audio(audio_in):
        sample_rate, tensor = audio_in
        audio = torch.tensor(tensor)
        k = torch.from_numpy(tensor).float()
        audio = torchaudio.functional.resample(k, orig_freq=sample_rate, new_freq=model.config.sampling_rate)
        # process input
        audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate=model.config.sampling_rate).to(DEVICE)
        # generate translation
        if MODEL_TYPE == "S2TT":
            output_tokens = model.generate(**audio_inputs, tgt_lang="eng")
            translated_text_from_audio = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        else:
            output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
            translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        print(f"Translation from audio: {translated_text_from_audio}")
        return translated_text_from_audio

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Speech to translated text

    1. Record your voice in your favorite language 
    2. Get the text in english translated

    ## References

    * UI created with [Gradio](https://www.gradio.app/)
    * Speech to text using [Meta Seemless V2](https://ai.meta.com/research/seamless-communication/) and more particularly [SeamlessM4T v2](https://ai.meta.com/resources/models-and-libraries/seamless-communication-models/#seamlessm4t)
    """)
    input_audio = gr.Audio(sources=["microphone"])
    output_text = gr.Textbox(
        label="Translated Text", 
        value='...translated text...', 
        interactive=True,
        show_copy_button=True, 
        show_label=True)
    input_audio.change(reverse_audio, input_audio, output_text)

if __name__ == "__main__":
    demo.launch(share=True)
