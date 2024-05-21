from transformers import AutoProcessor, SeamlessM4Tv2Model
import scipy
from pydub import AudioSegment


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")


# TODOS
# Concatenate file with overlay : https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentoverlay
# Use the specific seamless model for Text to Speach

# Below is commented as mms-tts gave bad results, and speaker cannot be choosen
"""
model_vits = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
"""
input_text_list  = [[0, "The system engineer wants to verify certain requirements from the hydrogen aircraft specification."],
                    [10, "He just selects the requirements, and creates a new verification request from this selection."],
                    [20, "The verification request will gather the parameters corresponding to the system design parameters or the performance criteria to meet."],
                    [30, "Using the gateway, the simulation analysis will map the simulation parameters with the parameters from the verification request."],
                    [40, "Once the mapping is done, the simulation model is sent to be lifecycle managed by Teamcenter."],
                    [50, "The simulation analyst can then run the verification activity using this tool and provide the requirement satisfaction status."],
                    [60, "After execution, the simulation results are automatically sent to that other tool."],
                    [70, "All parametric requirements are automatically rolled up." ],
                    [80, "System engineer can change some design parameters' value to evaluate the impact on preformances and related requirements."]]

#languages = ["fra","cmn","deu"]
languages = ["fra","cmn"]

"""
Language list:
afr,amh,arb,ary,arz,asm,azj,bel,ben,bos,bul,cat,ceb,ces,ckb,cmn,cmn_Hant,cym,dan,deu,ell,eng,est,eus,fin,fra,fuv,gaz,gle,glg,guj,heb,hin,hrv,hun,hye,ibo,ind,isl,ita,jav,
jpn,kan,kat,kaz,khk,khm,kir,kor,lao,lit,lug,luo,lvs,mai,mal,mar,mkd,mlt,mni,mya,nld,nno,nob,npi,nya,ory,pan,pbt,pes,pol,por,ron,rus,sat,slk,slv,sna,snd,som,spa,srp,swe,
swh,tam,tel,tgk,tgl,tha,tur,ukr,urd,uzn,vie,yor,yue,zlm,zul.
"""

speaker_ids= [7,8] #7 gives a good sound output in english. other are are very metalic

combined_sounds_eng = AudioSegment.silent(duration=100000)
for speaker in speaker_ids:
    for time, input_text_str in input_text_list:
        text_inputs = processor(text = input_text_str, src_lang="eng", return_tensors="pt")
        audio_from_raw_text = model.generate(**text_inputs, tgt_lang="eng", speaker_id=speaker)[0].cpu().numpy().squeeze()
        scipy.io.wavfile.write("raw_input_with_SeamlessM4Tv2Model.wav", rate=model.config.sampling_rate, data=audio_from_raw_text)
        sound = AudioSegment.from_file("raw_input_with_SeamlessM4Tv2Model.wav", format="wav")
        combined_sounds_eng = combined_sounds_eng.overlay(sound, position=time*1000)
    combined_sounds_eng.export(f"eng_all_{speaker}.wav", format="wav")



for speaker in speaker_ids:
    for lang in languages:
        combined_sounds_lang = AudioSegment.silent(duration=100000)
        for time, input_text_str in input_text_list:
            text_inputs = processor(text = input_text_str, src_lang="eng", return_tensors="pt")
            audio_array_from_text = model.generate(**text_inputs, tgt_lang=lang, speaker_id=speaker)[0].cpu().numpy().squeeze()
            scipy.io.wavfile.write("translation_with_SeamlessM4Tv2Model.wav", rate=model.config.sampling_rate, data=audio_array_from_text)
            sound = AudioSegment.from_file("translation_with_SeamlessM4Tv2Model.wav", format="wav")
            combined_sounds_lang = combined_sounds_lang.overlay(sound, position=time*1000)
        combined_sounds_lang.export(f"{lang}_all_{speaker}.wav", format="wav")

    """
    # see that : https://huggingface.co/facebook/mms-tts-fra
    # this is the recommanded 
    inputs = tokenizer(input_text_str, return_tensors="pt")
    
    with torch.no_grad():
        output = model_vits(**inputs).waveform
    scipy.io.wavfile.write(f"raw_input_with_mm_tts_eng_{i}.wav", rate=model_vits.config.sampling_rate, data=output.cpu().float().numpy().T)
    """
