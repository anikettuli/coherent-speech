import os

from faster_whisper import WhisperModel
from pydub import AudioSegment
import ffmpeg
import tempfile

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
                .output(output_audio_path, ac=1, ar="16000") # 16kHz mono for whisper
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr)
            raise e

    def run_asr(self, audio_path, model_size="small"):
        print(f"Running ASR (faster-whisper, {model_size})...")
        device = "cuda" if self.tts.has_gpu() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        extracted_segments = []
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            extracted_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        return extracted_segments

    def synthesize_and_align(self, segments, video_duration, tts_model, kokoro_voice="af_heart"):
        print("Synthesizing speech and aligning...")
        output_audio = AudioSegment.silent(duration=int(video_duration * 1000) + 2000) # Give buffer
        temp_dir = tempfile.mkdtemp()
        
        for idx, seg in enumerate(segments):
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
