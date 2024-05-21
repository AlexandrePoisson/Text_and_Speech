# Text <-> Speech Experiments

## Installation

Installation provided for a Linux env. To install on Windows, use the WSL.

Create a venv and install the required modules

    alexandre:~$ python3 -m venv venv_Seamless
    alexandre:~$ source venv_Seamless/bin/activate
    (venv_Seamless) alexandre:~$pip install torchaudio gradio sentencepiece protobuf scipy

For text to speech example:

Other python libs are required

    pip install scipy ffmpeg

You might also need ffmepg:

    sudo apt update
    sudo apt install ffmpeg

Alternative is to install all Python libs using the provided requirement file.

    pip install -r requirements.txt

## Runnning

### Speech to translated text

To run the speech to translated text

    python3 Speech_to_translated_text.py

Open a browser and copy paste the link that you get when executing the previous command (likely http://127.0.0.1:7860/)

Then you can use the Gradio user interface to record your speech in any language and then click Submit to generate the translated text into english.


https://github.com/AlexandrePoisson/Text_and_Speech/assets/13329302/f1f90084-44c2-4661-a0aa-0e33107a0199


### Text to speech

Open Text_to_speech.py file and edit the text and the timings

    python3 Text_to_speech.py

Then different wav files are generated containing the text.
