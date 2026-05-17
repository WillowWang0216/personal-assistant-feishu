#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from faster_whisper import WhisperModel

print("Loading model...", flush=True)
model = WhisperModel('base', device='cpu', compute_type='int8')

print("Transcribing...", flush=True)
segments, info = model.transcribe(
    r'C:\Users\xlab_w\.nanobot\media\om_x100b6f3c11146490c27c1c1bf4cce3f_产品经理.m4a',
    language='zh'
)

print("Saving transcript...", flush=True)
lines = []
for seg in segments:
    lines.append(seg.text)

with open(r'C:\Users\xlab_w\.nanobot\workspace\transcript_raw.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(lines))

print("DONE", flush=True)
