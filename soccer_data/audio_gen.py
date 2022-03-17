#/usr/bin/env python

import hashlib
from pathlib import Path

import boto3
from pydub import AudioSegment
from pydub.playback import play


frame_rate = 16000
channels = 1
sample_width = 2 # 16-bit


class Polly:
    def __init__(self, profile, region, cache_dir=None):
        session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )
        self.polly = session.client('polly')
        self.cache_dir = cache_dir and Path(cache_dir).resolve()
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text):
        pcm = None
        if self.cache_dir:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            cache_file = self.cache_dir / text_hash
            if cache_file.exists():
                pcm = cache_file.read_bytes()

        if pcm is None:
            response = self.polly.synthesize_speech(
                VoiceId='Takumi',
                Engine = 'neural',
                OutputFormat='pcm',
                Text=text,
            )
            pcm = response['AudioStream'].read()
            if self.cache_dir:
                cache_file.write_bytes(pcm)

        return AudioSegment(
            data=pcm,
            channels=channels,
            sample_width=sample_width, # 16 bit
            frame_rate=frame_rate,
        )


class AudioGen:
    def __init__(self, duration, polly=None):
        self.polly = polly or Polly()
        self.canvas = AudioSegment.silent(frame_rate=frame_rate, duration=int(duration * 1000))

    def add(self, when, audio):
        self.canvas = self.canvas.overlay(audio, position=int(when * 1000))

    def get(self):
        return self.canvas
