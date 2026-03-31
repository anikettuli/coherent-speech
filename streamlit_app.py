import os
import warnings
import tempfile
import traceback
import streamlit as st
from tts_manager import TTSManager
from pipeline import AudioVideoPipeline

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Coherent Speech Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_managers():
    manager = TTSManager()
    pipe = AudioVideoPipeline(manager)
    return manager, pipe

tts_manager, pipeline = get_managers()

st.title("🎙️ Coherent Speech")
st.caption("`[ENGINE: KOKORO-82M] | [ASR: WHISPER] | [RESTORE DAMAGED SPEECH]`")
st.markdown("**Enterprise-grade AI tool to automatically transcribe, clean, and restore damaged, garbled, or low-volume audio.**")
st.divider()

kokoro_voices = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", 
    "am_puck", "am_michael", "am_adam", "bf_alice", "bf_emma", 
    "bm_george", "bm_lewis"
]

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("1. Source Input")
        uploaded_video = st.file_uploader("Upload damaged audio/video (noisy, garbled, low volume)", type=["mp4", "mkv", "mov", "wav", "mp3"])
        
    with st.container(border=True):
        st.subheader("2. Generation Config")
        
        has_gpu = tts_manager.has_gpu()
        hw_ops = ["Auto"]
        if has_gpu:
            gpu_name = "GPU (Apple Silicon M-Series)" if getattr(tts_manager, "device", "") == "mps" else "GPU (CUDA)"
            hw_ops.append(gpu_name)
        hw_ops.append("CPU")
        
        selected_hw = st.radio("Hardware Acceleration:", hw_ops, horizontal=True)
        
        voice_mode = st.radio(
            "Synthesis Strategy:",
            ["[MODELS] Built-in Studio Voices", "[CLONING] Pipeline (In-Dev)"],
            horizontal=False
        )
        
        if voice_mode == "[MODELS] Built-in Studio Voices":
            selected_voice = st.selectbox("Select Voice Profile:", kokoro_voices)
            cloning_active = False
        else:
            st.success("=> [HOOK ENGAGED]: Auto-Extraction Active. AI will isolate a reference voice sample from the file.")
            selected_voice = "Custom Clone"
            cloning_active = True
            
        whisper_size = st.select_slider(
            "ASR Precision Target:",
            options=["tiny", "base", "small", "medium", "large"],
            value="medium"
        )

with col2:
    with st.container(border=True):
        st.subheader("3. Execution Terminal")
        
        if uploaded_video:
            # We can only show video if it's a video file, otherwise audio
            if uploaded_video.name.endswith(("mp4", "mov", "mkv")):
                st.video(uploaded_video)
            else:
                st.audio(uploaded_video)
            
        if 'cancel_pipeline' not in st.session_state:
            st.session_state.cancel_pipeline = False

        col_exec, col_stop = st.columns([3, 1])
        with col_exec:
            launch = st.button("▶ EXECUTE RESTORATION", use_container_width=True)
        with col_stop:
            if st.button("⏹ STOP", use_container_width=True, type="secondary"):
                st.session_state.cancel_pipeline = True

        if launch:
            st.session_state.cancel_pipeline = False
            if not uploaded_video:
                st.error("ERR: No source media detected.")
            else:
                progress_bar = st.progress(0, text="Initializing Pipeline...")
                status_console = st.empty()
                status_console.code("$> init execution\n[OK]")
                
                def update_prog(pct, txt):
                    progress_bar.progress(pct, text=txt)
                    
                def check_prog():
                    return st.session_state.cancel_pipeline

                try:
                    if uploaded_video.size > 500 * 1024 * 1024:
                        raise ValueError("File exceeds maximum limit of 500MB. Please use a smaller file or chunk the audio.")
                    update_prog(5, "Extracting Audio/Preparing File...")
                    
                    with tempfile.TemporaryDirectory() as base_temp:
                
                        temp_vid_path = os.path.join(base_temp, uploaded_video.name)
                        temp_audio_path = os.path.join(base_temp, "extracted_audio.wav")
                        final_out_path = os.path.join(base_temp, "restored_media.mp4")
                        
                        with open(temp_vid_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())
                            
                        pipeline.extract_audio(temp_vid_path, temp_audio_path)
                        
                        if not st.session_state.cancel_pipeline:
                            update_prog(10, "Starting Whisper ASR Engine...")
                            segments = pipeline.run_asr(
                                temp_audio_path, 
                                model_size=whisper_size,
                                progress_callback=update_prog,
                                check_cancel=check_prog,
                                device_override=selected_hw
                            )
                            
                        if not segments and not st.session_state.cancel_pipeline:
                            raise ValueError("No speech detected in the uploaded file.")
                        
                        if not st.session_state.cancel_pipeline:
                            video_dur = pipeline.get_video_duration(temp_vid_path)
                            
                            if cloning_active:
                                composed_path = pipeline.synthesize_and_align_cloning(
                                    segments, video_dur, progress_callback=update_prog, check_cancel=check_prog
                                )
                            else:
                                composed_path = pipeline.synthesize_and_align(
                                    segments, video_dur, "Kokoro Modern", kokoro_voice=selected_voice,
                                    progress_callback=update_prog, check_cancel=check_prog
                                )
                        
                        if not st.session_state.cancel_pipeline:
                            update_prog(90, "Multiplexing Final Matrix...")
                            # If they uploaded audio, just give them the composed clean audio back
                            if uploaded_video.name.endswith(("wav", "mp3", "m4a")):
                                final_path = composed_path
                                mime_type = "audio/wav"
                                dl_name = "Restored_Audio.wav"
                            else:
                                final_path = pipeline.assemble_final_video(
                                    temp_vid_path, composed_path, final_out_path
                                )
                                mime_type = "video/mp4"
                                dl_name = "Restored_Media.mp4"
                            
                            update_prog(100, "RESTORATION COMPLETE ✓")
                            status_console.code("$> pipeline_complete 100%\n[OK]")
                            
                            if mime_type.startswith("video"):
                                st.video(final_path)
                            else:
                                st.audio(final_path)
                            
                            with open(final_path, "rb") as file:
                                st.download_button(
                                    label="DOWNLOAD RESTORED MEDIA",
                                    data=file,
                                    file_name=dl_name,
                                    mime=mime_type,
                                    use_container_width=True
                                )
                        else:
                            st.warning("PIPELINE HALTED BY USER")
                            update_prog(0, "Halted.")
                            status_console.code("$> SIGINT Received. Terminated.\n[WARN]")
                
                except ValueError as ve:
                    st.error(f"Validation Error: {str(ve)}")
                except RuntimeError as re:
                    error_msg = str(re).lower()
                    if "out of memory" in error_msg or "cuda out of memory" in error_msg:
                        st.error("Hardware Error: GPU Out of Memory. Please try checking the 'CPU' fallback option or a smaller ASR model.")
                    else:
                        st.error(f"Runtime Fault: {str(re)}")
                except Exception as e:
                    st.error(f"PIPELINE PAULT: Failed to process file. \n\n{str(e)}")
                    print(traceback.format_exc())
