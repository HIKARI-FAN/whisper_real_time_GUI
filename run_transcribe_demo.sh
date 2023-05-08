#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python transcribe_demo.py --model medium --non_english --no_translation --whisper_source_lang ja
read -p "Press any key to continue..."