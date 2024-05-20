import os
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech

os.environ["GOOGLE_APPLICATION_CREDENTIALS_TRANSLATION"] = "C:\\Users\\Dell XPS White\\Desktop\\FYP\\psl-translator-c2a1a5fd7afb.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS_TTS"] = "C:\\Users\\Dell XPS White\\Desktop\\FYP\\psl-translator-5765ea9af21a.json"


translate_client = translate.Client()

text_to_translate = "Good"

target_language = 'ur'
translation = translate_client.translate(text_to_translate, target_language=target_language)

translated_text = translation['translatedText']

client = texttospeech.TextToSpeechClient()
synthesis_input = texttospeech.SynthesisInput(text=translated_text)

voice = texttospeech.VoiceSelectionParams(
    language_code="ur-PK",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

output_file = "output_urdu3.mp3"
with open(output_file, "wb") as out:
    out.write(response.audio_content)

print("Text translated to Urdu and speech synthesized. Output saved as:", output_file)
