
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

    def generate_speech_f5(self, text, output_path, ref_audio_path, ref_text):
        import subprocess
        import os
        import tempfile
        import shutil
        
        print(f"Synthesizing '{text[:30]}...' with F5-TTS")
        
        temp_dir = tempfile.mkdtemp()
        try:
            cmd = [
                "f5-tts_infer-cli",
                "--model", "F5TTS_Base",
                "--ref_audio", ref_audio_path,
                "--ref_text", ref_text,
                "--gen_text", text,
                "--output_dir", temp_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"F5-TTS Error: {result.stderr}")
                return False
                
            generated_file = os.path.join(temp_dir, "out.wav")
            if os.path.exists(generated_file):
                shutil.move(generated_file, output_path)
                return True
            else:
                wavs = [f for f in os.listdir(temp_dir) if f.endswith(".wav")]
                if wavs:
                    shutil.move(os.path.join(temp_dir, wavs[0]), output_path)
                    return True
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
