import librosa
import soundfile as sf
# import torch
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
from deep_translator import GoogleTranslator
# Set a cache directory where the model will be stored after download
# cache_dir = "./model"

# Load the processor and model using the cache directory
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
import os
import zipfile
import requests
from pocketsphinx import Decoder

# Define the cache directory
base_dir = os.path.dirname(__file__)  # Get the current script directory
cache_dir = os.path.join(base_dir, 'pocketsphinx_cache', 'en-us')
# cache_dir = '\pocketsphinx_cache\en-us'
model_dir = cache_dir
dictionary_path = os.path.join(cache_dir, 'cmudict-en-us.dict')
# zip_url = "https://github.com/cmusphinx/pocketsphinx/blob/master/model/en-us/en-us.zip?raw=true"
# model_dir = os.path.join(os.path.dirname(__file__), 'pocketsphinx/model/en-us')
# Path to the dictionary file (adjust if necessary)
# dictionary_path = os.path.join(model_dir, 'cmudict-en-us.dict')


def translate_text(text: str, target_lang: str) -> str:
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated_text
    except Exception as e:
        return f"Error during translation: {str(e)}"

def preprocess_audio(file_path, target_sr=16000):
    """Load and normalize the audio"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = librosa.util.normalize(audio)
    return audio, sr

def audio_to_text(audiopath,target_lang):

    audio, sr = preprocess_audio(audiopath)

    temp_audio_path = 'temp_audio.wav'
    sf.write(temp_audio_path, audio, sr)

    config = Decoder.default_config()
    config.set_string('-hmm', model_dir)  # Path to acoustic model
    config.set_string('-dict', dictionary_path)  # Path to dictionary
    print(temp_audio_path)
    # Start decoding from an audio file
    decoder = Decoder(config)
    decoder.start_utt()
# Read audio file in chunks
    with open(temp_audio_path, 'rb') as audio_file:
        while True:
            buf = audio_file.read(1024)  # Read audio in chunks
            if not buf:
                break
            decoder.process_raw(buf, False, False)
    decoder.end_utt()

# Get recognized text
    transcription = decoder.hyp().hypstr
    translated_transcription = translate_text(transcription,target_lang)

    os.remove(temp_audio_path)
    return translated_transcription













# def translate_text(text: str, target_lang: str) -> str:
#     try:
#         translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
#         return translated_text
#     except Exception as e:
#         return f"Error during translation: {str(e)}"

# def preprocess_audio(file_path, target_sr=16000):
#     """Load and normalize the audio"""
#     audio, sr = librosa.load(file_path, sr=target_sr)
#     audio = librosa.util.normalize(audio)
#     return audio, sr

def speech_to_text(file_path,target_lang):
    """Convert speech to text"""
    # Load and preprocess the audio
    audio, sr = preprocess_audio(file_path)
   
    # Prepare the audio input for the model using the processor
    # input_values = processor(audio, return_tensors="pt", sampling_rate=sr).input_values
   
    # Perform the transcription
    # with torch.no_grad():
        # logits = model(input_values).logits
   
    # Get the predicted token IDs and decode them to text
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.decode(predicted_ids[0])
    
    # translated_transcription = translate_text(transcription,target_lang)
    # return translated_transcription

def clean_transcription(text):
    """Clean the transcription by removing unwanted characters"""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

# user_input
# # Specify the file path for the audio
# file_path = "OSR_us_000_0061_8k.wav"

# # Run the transcription process
# transcription = speech_to_text(file_path)
# cleaned_transcription = clean_transcription(transcription)

# print(f"Original Transcription: {transcription}")
# print(f"Cleaned Transcription: {cleaned_transcription}")


# def translate_into_text(file_path):
#     transcription = speech_to_text(file_path)
#     cleaned_transcription = clean_transcription(transcription)
#     return cleaned_transcription

# from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# import torch
# import librosa
# import numpy as np
# import re
# def preprocess_audio(file_path,target_sr = 16000):

#     audio, sr = librosa.load(file_path,sr = target_sr)

#     audio = librosa.util.normalize(audio)

#     return audio, sr

# def extract_mfcc(audio,sr,n_mfcc = 13):
#     mfcc = librosa.feature.mfcc(y = audio,sr = sr,n_mfcc = n_mfcc)
#     mfcc = mfcc.T
#     return mfcc

# tokenizer = Wav2VecTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# model = wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# def speech_to_text(file_path):
#     audio,sr = preprocess_audio(file_path)

#     input_values = tokenizer(audio,return_tensors="pt",padding="longest").input_values
    
#     with torch.no_grad():
#           logits =model(input_values).logists

#     predicted_ids = torch.argmax(logits,dim=-1)      
#     transcription = tokenizer.decode(predicted_ids[0])
#     return transcription

# transcription = speech_to_text("")
# print(transcription)


# def clean_transcription(text):
#      text = re.sub(r'[^a-zA-Z0-9\s]','',text)
#      text = text.lower()

#      return text

# cleaned_text = clean_transcription(transcription)
# print(cleaned_text)

