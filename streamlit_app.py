import streamlit as st
from tts_manager import TTSManager
from pipeline import AudioVideoPipeline


st.set_page_config(
    page_title="Coherant | Dev Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def get_managers():
    manager = TTSManager()
    pipe = AudioVideoPipeline(manager)
    return manager, pipe

tts_manager, pipeline = get_managers()

st.title("🗣️ COHERANT-SPEECH")
st.caption("`[ENGINE: KOKORO-82M] | [ASR: WHISPER] | [RUNTIME: NATIVE]`")
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
        uploaded_video = st.file_uploader("Upload lecture video (mp4, mkv)", type=["mp4", "mkv", "mov"])
        
    with st.container(border=True):
        st.subheader("2. Generation Config")
        voice_mode = st.radio(
            "Synthesis Strategy:",
            ["[MODELS] Built-in Studio Voices", "[CLONING] Pipeline (In-Dev)"],
            horizontal=False
        )
        
        if voice_mode == "[MODELS] Built-in Studio Voices":
            selected_voice = st.selectbox("Select Voice Profile:", kokoro_voices)
            cloning_active = False
        else:
            st.success("=> [HOOK ENGAGED]: Auto-Extraction Active. AI will isolate a 5-second slice from the uploaded video for voice map extraction.")
            selected_voice = "Custom Clone"
            cloning_active = True
            
        whisper_size = st.select_slider(
            "ASR Precision Target:",
            options=["tiny", "base", "small", "medium", "large"],
            value="small"
        )

with col2:
    with st.container(border=True):
        st.subheader("3. Execution Terminal")
        
        if uploaded_video:
            st.video(uploaded_video)
            
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
                    update_prog(5, "Extracting Audio from Video...")
                    
                    temp_vid_path = "temp_source.mp4"
                    with open(temp_vid_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                        
                    pipeline.extract_audio(temp_vid_path, "temp_audio.wav")
                    
                    if not st.session_state.cancel_pipeline:
                        update_prog(10, "Starting Whisper ASR Engine...")
                        segments = pipeline.run_asr(
                            "temp_audio.wav", 
                            model_size=whisper_size,
                            progress_callback=update_prog,
                            check_cancel=check_prog
                        )
                    
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
                        final_path = pipeline.assemble_final_video(
                            temp_vid_path, composed_path, "output_restored_video.mp4"
                        )
                        
                        update_prog(100, "RESTORATION COMPLETE ✓")
                        status_console.code("$> pipeline_complete 100%\n[OK]")
                        st.video(final_path)
                        
                        dl_name = "Cloned_Lecture.mp4" if cloning_active else "Studio_Lecture.mp4"
                        with open(final_path, "rb") as file:
                            st.download_button(
                                label="DOWNLOAD MEDIA",
                                data=file,
                                file_name=dl_name,
                                mime="video/mp4",
                                use_container_width=True
                            )
                    else:
                        st.warning("PIPELINE HALTED BY USER")
                        update_prog(0, "Halted.")
                        status_console.code("$> SIGINT Received. Terminated.\n[WARN]")
                except Exception as e:
                    st.error(f"EXCEPTION FAULT: {str(e)}")
