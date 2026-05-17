from faster_whisper import WhisperModel
import os

audio_path = r'D:\Work\project\nanobot-feishu-specilized\workspace\会议录音.m4a'
output_path = r'D:\Work\project\nanobot-feishu-specilized\workspace\会议录音.txt'

print('Starting transcription...')
model = WhisperModel('base', device='cpu', compute_type='int8')
print('Model loaded')

segments, info = model.transcribe(audio_path, language='zh')

print(f'Language: {info.language}, Duration: {info.duration:.1f}s')
print('Transcribing...')

with open(output_path, 'w', encoding='utf-8') as f:
    for segment in segments:
        line = f'[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}'
        f.write(line + '\n')
        print(line)

print(f'\nTranscription complete! Saved to: {output_path}')
