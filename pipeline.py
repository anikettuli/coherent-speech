import os

from faster_whisper import WhisperModel
from pydub import AudioSegment
import ffmpeg
import tempfile
import hashlib
import json
from pathlib import Path

class AudioVideoPipeline:
    def __init__(self, tts_manager):
        self.tts = tts_manager
    
    def get_video_duration(self, video_path):

        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])

    def extract_audio(self, video_path, output_audio_path):
        print("Extracting audio from video...")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_audio_path, ac=1, ar="16000", threads=16) # use all 16 threads of 5900HS
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr)
            raise e

    def run_asr(self, audio_path, model_size="small", progress_callback=None, check_cancel=None, device_override=None):
        # 0. Check Cache
        cache_dir = Path(".cache/asr")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate file hash
        file_hash = hashlib.md5(open(audio_path, 'rb').read()).hexdigest()
        cache_file = cache_dir / f"{file_hash}_{model_size}.json"
        
        if cache_file.exists():
            print(f"CACHE HIT: Loading segments from {cache_file}")
            if progress_callback: progress_callback(25, "Loading Cached Transcription...")
            with open(cache_file, 'r') as f:
                return json.load(f)

        print(f"Running ASR (faster-whisper, {model_size})...")
        if progress_callback:
            progress_callback(10, "Loading Whisper ASR Engine...")
        
        device = "cuda" if self.tts.has_gpu() else "cpu"
        if device_override == "GPU (CUDA)":
            device = "cuda"
        elif device_override == "CPU":
            device = "cpu"
            
        compute_type = "float16" if device == "cuda" else "int8"
        
        try:
            model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type,
                download_root="models/whisper",
                cpu_threads=16,
                num_workers=4
            )
            # Trigger library check early
            segments, info = model.transcribe(
                audio_path, 
                beam_size=2,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        except Exception as e:
            if "libcublas" in str(e) or "libcudnn" in str(e) or "RuntimeError" in str(e):
                print("CUDA libs missing or failing, falling back to robust CPU ASR...")
                model = WhisperModel(
                    model_size, 
                    device="cpu", 
                    compute_type="int8",
                    download_root="models/whisper",
                    cpu_threads=16,
                    num_workers=4
                )
                segments, info = model.transcribe(
                    audio_path, 
                    beam_size=2,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
            else:
                raise e
        
        extracted_segments = []
        for i, segment in enumerate(segments):
            if check_cancel and check_cancel():
                return []
            if progress_callback and (i % 3 == 0):
                progress_callback(min(25, 10 + i), f"Transcribing timeline [{segment.start:.1f}s]...")
                
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            extracted_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            
        # 5. Store Cache
        with open(cache_file, 'w') as f:
            json.dump(extracted_segments, f)
            
        return extracted_segments

    def synthesize_and_align(self, segments, video_duration, tts_model, kokoro_voice="af_heart", progress_callback=None, check_cancel=None):
        print("Synthesizing speech and aligning...")
        output_audio = AudioSegment.silent(duration=int(video_duration * 1000) + 2000) # Give buffer
        temp_dir = tempfile.mkdtemp()
        
        for idx, seg in enumerate(segments):
            if check_cancel and check_cancel():
                break
            if progress_callback:
                pct = 30 + int(60 * (idx / len(segments)))
                progress_callback(pct, f"Synthesizing Kokoro audio ({idx+1}/{len(segments)})...")
                
            text = seg["text"].strip()
            if not text:
                continue
                
            seg_wav_path = os.path.join(temp_dir, f"seg_{idx}.wav")
            
            # Generate TTS
            self.tts.generate_speech_kokoro(text, seg_wav_path, voice=kokoro_voice)
                
            if os.path.exists(seg_wav_path):
                # Load generated audio
                tts_clip = AudioSegment.from_wav(seg_wav_path)
                # Exact original start time
                position_ms = int(seg["start"] * 1000)
                output_audio = output_audio.overlay(tts_clip, position=position_ms)
        
        composed_path = "composed_clean_audio.wav"
        output_audio.export(composed_path, format="wav")
        return composed_path

    def extract_reference(self, segments, audio_path):
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(audio_path)
        
        best_seg = None
        for seg in segments:
            dur = seg["end"] - seg["start"]
            if 4.0 <= dur <= 15.0:
                best_seg = seg
                break
                
        if not best_seg and len(segments) > 0:
            best_seg = segments[0]
            
        if not best_seg:
            raise ValueError("No speech segments found to extract reference from.")
            
        start_ms = int(max(0, best_seg["start"]) * 1000)
        end_ms = int(best_seg["end"] * 1000)
        
        ref_clip = audio[start_ms:end_ms]
        ref_path = "f5_reference.wav"
        ref_clip.export(ref_path, format="wav")
        return ref_path, best_seg["text"].strip()

    def synthesize_and_align_cloning(self, segments, video_duration, progress_callback=None, check_cancel=None):
        print("Synthesizing cloned speech and aligning...")
        if progress_callback:
            progress_callback(30, "Extracting Cloned Voice DNA...")
        from pydub import AudioSegment
        import concurrent.futures
        import threading
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            ctx = get_script_run_ctx()
        except Exception:
            ctx = None
            
        output_audio = AudioSegment.silent(duration=int(video_duration * 1000) + 2000)
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        
        ref_path, ref_text = self.extract_reference(segments, "temp_audio.wav")
        print(f"Extracted Reference: {ref_text}")
        
        def process_segment(idx, seg):
            if ctx is not None:
                add_script_run_ctx(threading.current_thread(), ctx)
                
            text = seg["text"].strip()
            if not text:
                return None
                
            seg_wav_path = os.path.join(temp_dir, f"seg_{idx}.wav")
            success = self.tts.generate_speech_f5(text, seg_wav_path, ref_path, ref_text)
            
            if success and os.path.exists(seg_wav_path):
                tts_clip = AudioSegment.from_wav(seg_wav_path)
                position_ms = int(seg["start"] * 1000)
                return (tts_clip, position_ms)
            return None

        completed = 0
        total = len(segments)
        
        # Dispatch parallel TTS generation to saturate the GPU (RTX 3070 has 8GB VRAM)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_segment, idx, seg): idx for idx, seg in enumerate(segments)}
            
            for future in concurrent.futures.as_completed(futures):
                if check_cancel and check_cancel():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
                result = future.result()
                if result:
                    tts_clip, position_ms = result
                    output_audio = output_audio.overlay(tts_clip, position=position_ms)
                    
                completed += 1
                if progress_callback: 
                    pct = 35 + int(55 * (completed / max(1, total)))
                    progress_callback(pct, f"Synthesizing F5 Clone Audio ({completed}/{total})...")
        
        composed_path = "composed_clean_audio.wav"
        output_audio.export(composed_path, format="wav")
        return composed_path

    def assemble_final_video(self, original_video_path, new_audio_path, final_video_path):
        print("Assembling final video...")
        try:
            video_input = ffmpeg.input(original_video_path)
            audio_input = ffmpeg.input(new_audio_path)
            
            (
                ffmpeg
                .output(video_input.video, audio_input.audio, final_video_path, vcodec="copy", acodec="aac")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr)
            raise e
        return final_video_path
