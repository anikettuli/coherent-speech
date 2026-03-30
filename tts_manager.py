
import torch
import numpy as np
import soundfile as sf


class TTSManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kokoro_pipeline = None

    def has_gpu(self):
        return self.device == "cuda"
    
    def init_kokoro(self, lang_code='a'):
        if self.kokoro_pipeline is None:
            from kokoro import KPipeline
            print(f"Loading Kokoro-82M on {self.device}...")
            self.kokoro_pipeline = KPipeline(lang_code=lang_code)

    def generate_speech_kokoro(self, text, output_path, voice='af_heart'):
        if not self.kokoro_pipeline:
            self.init_kokoro()
            
        print(f"Synthesizing '{text[:30]}...' with Kokoro ({voice})")
        generator = self.kokoro_pipeline(
            text, voice=voice, speed=1.0, split_pattern=r'\n+'
        )
        
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                all_audio.append(audio)
            
        if all_audio:
            full_audio = np.concatenate(all_audio)
            sf.write(output_path, full_audio, 24000)
            return True
        return False
