
import torch
import numpy as np
import soundfile as sf


class TTSManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kokoro_pipeline = None
        import threading
        self.f5_lock = threading.Lock()
        self.f5_ema_model = None

    def has_gpu(self):
        return self.device == "cuda"
    
    def init_kokoro(self, lang_code='a'):
        if self.kokoro_pipeline is None:
            from kokoro import KPipeline
            print(f"Loading Kokoro-82M on {self.device}...")
            self.kokoro_pipeline = KPipeline(lang_code=lang_code, device=self.device)

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

    def init_f5(self):
        print(f"Loading F5TTS_Base on {self.device} into memory...")
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from importlib.resources import files
        from omegaconf import OmegaConf
        from hydra.utils import get_class
        from cached_path import cached_path
        
        self.f5_vocoder = load_vocoder(
            vocoder_name="vocos", is_local=False, local_path="", device=self.device
        )
        
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        
        ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
        
        self.f5_ema_model = load_model(
            model_cls, model_cfg.model.arch, ckpt_file, mel_spec_type="vocos", vocab_file="", device=self.device
        )

    def generate_speech_f5(self, text, output_path, ref_audio_path, ref_text):
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        import soundfile as sf
        
        import f5_tts.infer.utils_infer as utils_infer
        import concurrent.futures
        
        with self.f5_lock:
            if self.f5_ema_model is None:
                self.init_f5()
                
            print(f"Synthesizing '{text[:30]}...' with native F5-TTS")
            
            # Monkey-patch F5-TTS to prevent its internal ThreadPoolExecutor
            # from running multiple inference threads concurrently on the same
            # non-thread-safe transformer model state.
            original_executor = utils_infer.ThreadPoolExecutor
            utils_infer.ThreadPoolExecutor = lambda *args, **kwargs: concurrent.futures.ThreadPoolExecutor(max_workers=1)
            
            try:
                ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
                audio_segment, final_sample_rate, _ = infer_process(
                    ref_audio,
                    ref_text,
                    text,
                    self.f5_ema_model,
                    self.f5_vocoder,
                    mel_spec_type="vocos",
                    target_rms=0.1,
                    cross_fade_duration=0.15,
                    nfe_step=16,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1.0,
                    speed=1.0,
                    fix_duration=None,
                    device=self.device,
                )
                sf.write(output_path, audio_segment, final_sample_rate)
                return True
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Native F5-TTS Generation Error: {str(e)}")
                return False
            finally:
                utils_infer.ThreadPoolExecutor = original_executor
