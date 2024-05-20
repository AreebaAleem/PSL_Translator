import os
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech

# Set the path to your service account key files
os.environ["GOOGLE_APPLICATION_CREDENTIALS_TRANSLATION"] = "C:\\Users\\Dell XPS White\\Desktop\\FYP\\psl-translator-c2a1a5fd7afb.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS_TTS"] = "C:\\Users\\Dell XPS White\\Desktop\\FYP\\psl-translator-5765ea9af21a.json"

# Initialize the translation client
translate_client = translate.Client()

# Text to translate
text_to_translate = "Good"

# Target language code
target_language = 'ur'

# Translate text
translation = translate_client.translate(text_to_translate, target_language=target_language)

# Extract translated text
translated_text = translation['translatedText']

# Initialize the Text-to-Speech client
client = texttospeech.TextToSpeechClient()

# Set up input text
synthesis_input = texttospeech.SynthesisInput(text=translated_text)

# Set up voice parameters for Urdu language
voice = texttospeech.VoiceSelectionParams(
    language_code="ur-PK",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Set up audio configuration
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Synthesize speech
response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

# Save the audio to a file
output_file = "output_urdu3.mp3"
with open(output_file, "wb") as out:
    out.write(response.audio_content)

print("Text translated to Urdu and speech synthesized. Output saved as:", output_file)
