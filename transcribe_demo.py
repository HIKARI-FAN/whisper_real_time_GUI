#! python3.7

import argparse
import io
import os
import sys
import speech_recognition as sr
import whisper
import torch
import deepl
import webrtcvad
import collections
import tkinter as tk

from datetime import datetime, timedelta
from pydub import AudioSegment
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--device", default="gpu", help="Device to use.", type=str)
    parser.add_argument("--deepl_auth_key", default="", help="Deepl API key.", type=str)
    parser.add_argument("--deepl_target_lang", default="zh", help="Deepl target language.", type=str)
    parser.add_argument("--whisper_source_lang", default="", help="Whisper source language.", type=str)
    parser.add_argument("--db_threshold", default=-50, help="Whisper db threshold.", type=float)
    parser.add_argument("--no_translation", action='store_true')
    parser.add_argument("--pop_up", action='store_true')

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Initialize the Deepl translator
    translator = None
    use_deepl = False
    if args.deepl_auth_key != "":
        translator = deepl.Translator(args.deepl_auth_key)
        use_deepl = True

    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
        
    # Load / Download model
    model = args.model
    if args.model != "large" and args.model != "large-v2" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    # Function to save window position
    def save_window_position():
        window_position = f"{root.winfo_x()}x{root.winfo_y()}"
        with open("window_position.txt", "w") as f:
            f.write(window_position)

    # Function to close the window and save its position
    def close_window(event=None):
        save_window_position()
        root.destroy()
        sys.exit(0)


    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

     # Function to adjust font size based on window size
    def adjust_font_size(event):
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set base font size and adjust it based on screen resolution
        base_font_size = 18
        font_size = int(base_font_size * min(screen_width, screen_height) / 1080)
        transcription_label.config(font=("Helvetica", font_size, "bold"))


    # Create a top-level window with a dark background
    root = tk.Tk()
    root.title("Subtittles")
    root.attributes("-topmost", True)
    root.configure(background="black")

    # Load the saved window position
    if os.path.exists("window_position.txt"):
        with open("window_position.txt", "r") as f:
            window_position = f.read()
        root.geometry(window_position)

     # Add a label to the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    base_font_size = 18
    font_size = int(base_font_size * min(screen_width, screen_height) / 1080)
    window_width = int(screen_width / 2)
    root.geometry(f"{window_width}x{font_size * 10}")
    transcription_label = tk.Label(root, text="", background="black", foreground="white", wraplength=window_width, justify="left", font=("Helvetica", font_size, "bold"))
    transcription_label.pack(padx=10, pady=10, expand=True, fill="both")



    # Bind the window resize event to adjust_font_size function
    root.bind("<Configure>", adjust_font_size)
    # Bind the close button and keyboard interrupt to close_window function
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.bind("<Control-c>", close_window)

    # Initialize speaker diarization
    vad = webrtcvad.Vad(3)  # Create a VAD object (3 = high aggressiveness)
    padding_duration_ms = 300
    chunk_duration_ms = 30
    num_padding_chunks = padding_duration_ms // chunk_duration_ms
    chunk_duration_bytes = chunk_duration_ms * source.SAMPLE_RATE * source.SAMPLE_WIDTH // 1000
    ring_buffer = collections.deque(maxlen=num_padding_chunks)
    triggered = False

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                 # Check for speaker change using WebRTC VAD
                is_speech = vad.is_speech(last_sample[-chunk_duration_bytes:], source.SAMPLE_RATE)
                if not triggered:
                    ring_buffer.append((last_sample, is_speech))
                    if len(ring_buffer) == num_padding_chunks and all([x[1] for x in ring_buffer]):
                        triggered = True
                        # Detected a speaker change
                        phrase_complete = True
                        print("(Speaker change detected!)")
                else:
                    if not is_speech:
                        triggered = False
                        ring_buffer.clear()

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Check the average decibel level
                audio_segment = AudioSegment.from_file(wav_data, format="wav")
                if audio_segment.dBFS < args.db_threshold:
                    continue  # Skip this audio segment if the average decibel level is below the threshold

                # Read the transcription.
                if args.whisper_source_lang != "":
                    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), language=args.whisper_source_lang)
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if text != "" and not args.no_translation:
                    if use_deepl:
                        # Translate the text using Deepl
                        translated_text = translator.translate_text(text, target_lang=args.deepl_target_lang)
                    else:
                        # Translate the text using whisper
                        result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), task="translate", language=args.whisper_source_lang)
                        translated_text = result['text'].strip()

                    text = f"{text} ({translated_text})"

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)

                else:
                    transcription[-1] = text


                # Only display the last two transcriptions
                displayed_transcription = transcription[-2:] if len(transcription) >= 2 else transcription

                # Update the transcription label text
                transcription_label.config(text="\n".join(displayed_transcription))

                # Update the tkinter window
                root.update()


                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.15)
        except KeyboardInterrupt:
            save_window_position()
            break
    
    root.mainloop()
    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()