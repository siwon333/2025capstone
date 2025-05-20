from pydub import AudioSegment
from pydub.generators import Sine
import random

def generate_audio(duration_ms=60000, silence_duration=2000):
    base = Sine(440).to_audio_segment(duration=duration_ms)
    silence = AudioSegment.silent(duration=silence_duration)
    positions = sorted(random.sample(range(5000, duration_ms - 5000, 5000), 10))

    output = AudioSegment.empty()
    prev = 0
    for pos in positions:
        output += base[prev:pos]
        output += silence
        prev = pos
    output += base[prev:]
    return output, positions
