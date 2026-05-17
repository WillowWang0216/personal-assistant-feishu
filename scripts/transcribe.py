#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transcribe audio file to text."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from faster_whisper import WhisperModel

def main():
    audio_path = r'C:\Users\xlab_w\.nanobot\media\om_x100b6f3c11146490c27c1c1bf4cce3f_产品经理.m4a'
    output_path = r'C:\Users\xlab_w\.nanobot\workspace\transcript_产品经理.txt'

    print("Loading model...", flush=True)
    model = WhisperModel('base', device='cpu', compute_type='int8')

    print("Transcribing...", flush=True)
    segments, info = model.transcribe(audio_path, language='zh')

    print(f"Detected language: {info.language}", flush=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"语言: {info.language}\n")
        f.write("=" * 50 + "\n\n")
        for i, seg in enumerate(segments, 1):
            f.write(f"[{i}] {seg.text}\n")

    print(f"Done! Saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()
