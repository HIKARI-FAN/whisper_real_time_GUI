@echo off
cd %~dp0
call env\Scripts\activate
python transcribe_demo.py --model medium --non_english --no_translation --whisper_source_lang ja
pause
