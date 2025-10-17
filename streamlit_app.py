"""
Streamlit Meeting Analyzer UI
Real-time meeting transcription and analysis with OpenAI + Grok.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import os
from audio_processor import AudioProcessor
from summarizer import MeetingSummarizer
import time
import json
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Meeting Analyzer",
    page_icon="🎤",
    layout="wide"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

if 'final_summary' not in st.session_state:
    st.session_state.final_summary = None

if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = None

if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None

if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# ============================================================================
# CALLBACK FUNCTIONS
# ============================================================================

def on_new_transcript(text: str):
    """Called by AudioProcessor when new transcript is ready."""
    st.session_state.transcript_text += text + " "
    
    if st.session_state.summarizer:
        analysis = st.session_state.summarizer.add_transcript(text)
        if analysis:
            st.session_state.current_analysis = analysis

def start_recording():
    """Start recording button callback."""
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")
    
    if not openai_key:
        st.error("❌ OPENAI_API_KEY not found in environment variables!")
        st.info("Set it with: export OPENAI_API_KEY='sk-your-key'")
        return
    
    if not xai_key:
        st.error("❌ XAI_API_KEY not found in environment variables!")
        st.info("Set it with: export XAI_API_KEY='xai-your-key'")
        return
    
    try:
        # Initialize components
        st.session_state.summarizer = MeetingSummarizer(xai_key)
        st.session_state.audio_processor = AudioProcessor(on_new_transcript, openai_key)
        
        # Clear previous data
        st.session_state.transcript_text = ""
        st.session_state.current_analysis = None
        st.session_state.final_summary = None
        st.session_state.start_time = datetime.now()
        
        # Start recording
        st.session_state.audio_processor.start_recording()
        st.session_state.is_recording = True
        
        st.success("✓ Recording started! Speak into your microphone.")
        
    except Exception as e:
        st.error(f"Error starting recording: {e}")
        import traceback
        st.code(traceback.format_exc())

def stop_recording():
    """Stop recording button callback."""
    if st.session_state.audio_processor:
        st.session_state.audio_processor.stop_recording()
    
    if st.session_state.summarizer:
        with st.spinner("Generating final summary..."):
            st.session_state.final_summary = st.session_state.summarizer.get_final_summary()
    
    st.session_state.is_recording = False
    st.success("✓ Recording stopped. Final summary generated.")

# ============================================================================
# UI LAYOUT
# ============================================================================

st.title("🎤 AI Meeting Analyzer")
st.markdown("""
Real-time meeting transcription and analysis powered by **OpenAI Realtime API** and **Grok**.

**Instructions:**
1. Click **Start Recording** to begin
2. Speak into your microphone for 5-7 minutes
3. Watch real-time transcript and insights appear
4. Click **Stop Recording** for final comprehensive summary

**Features:**
- ⚡ Ultra-low latency (<500ms)
- 🎯 Server-side Voice Activity Detection
- 🔇 Built-in noise reduction
- 🧠 Progressive summarization with Grok
""")

st.divider()

# Control buttons
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    if not st.session_state.is_recording:
        if st.button("▶️ Start Recording", type="primary", use_container_width=True):
            start_recording()
    else:
        st.button("▶️ Start Recording", disabled=True, use_container_width=True)

with col2:
    if st.session_state.is_recording:
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
            stop_recording()
    else:
        st.button("⏹️ Stop Recording", disabled=True, use_container_width=True)

# Recording indicator
if st.session_state.is_recording:
    duration = (datetime.now() - st.session_state.start_time).seconds
    st.markdown(f"""
    <div style='background-color: #ff4b4b; padding: 10px; border-radius: 5px; text-align: center;'>
        <b>🔴 RECORDING - {duration // 60}:{duration % 60:02d}</b>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Main content area
left_col, right_col = st.columns([1, 1])

# LEFT COLUMN: TRANSCRIPT
with left_col:
    st.subheader("📝 Live Transcript")
    
    transcript_placeholder = st.empty()
    
    if st.session_state.transcript_text:
        word_count = len(st.session_state.transcript_text.split())
        transcript_placeholder.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; max-height: 600px; overflow-y: auto;'>
            <p style='color: #666; font-size: 0.9em;'>{word_count} words</p>
            <p style='font-size: 1.1em; line-height: 1.6;'>{st.session_state.transcript_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        transcript_placeholder.info("Waiting for audio... Start speaking!")

# RIGHT COLUMN: INSIGHTS
with right_col:
    st.subheader("💡 Live Insights")
    
    if st.session_state.final_summary:
        st.success("✅ Meeting Complete - Final Summary")
        analysis = st.session_state.final_summary
    elif st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
    else:
        analysis = None
    
    if analysis:
        with st.expander("📋 Summary", expanded=True):
            st.write(analysis.get("summary", "No summary yet"))
        
        with st.expander("✅ Action Items", expanded=True):
            action_items = analysis.get("action_items", [])
            if action_items:
                for item in action_items:
                    st.markdown(f"- {item}")
            else:
                st.write("No action items identified yet")
        
        with st.expander("💻 IT Insights", expanded=False):
            it_insights = analysis.get("it_insights", [])
            if it_insights:
                for insight in it_insights:
                    st.markdown(f"- {insight}")
            else:
                st.write("No IT-specific insights yet")
        
        with st.expander("🎯 Key Decisions", expanded=False):
            decisions = analysis.get("key_decisions", [])
            if decisions:
                for decision in decisions:
                    st.markdown(f"- {decision}")
            else:
                st.write("No key decisions identified yet")
    else:
        st.info("Insights will appear after ~50 words of transcription")

# BOTTOM: STATISTICS
if st.session_state.summarizer:
    st.divider()
    st.subheader("📊 Statistics")
    
    stats = st.session_state.summarizer.get_stats()
    
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Duration", f"{stats['duration_minutes']} min")
    with stat_cols[1]:
        st.metric("Words", stats['word_count'])
    with stat_cols[2]:
        st.metric("Summaries", stats['summary_count'])
    with stat_cols[3]:
        st.metric("Segments", stats['transcript_segments'])

# FOOTER
st.divider()
st.caption("""
**Tech Stack:** OpenAI Realtime API (transcription) + Grok-4-Fast (analysis) + Streamlit (UI)  
**Cost per 10-min meeting:** ~$0.60 for transcription + $0.02 for analysis = **$0.62 total**  
**Features:** Server-side VAD, built-in noise reduction, <500ms latency
""")

# AUTO-REFRESH FOR LIVE UPDATES
if st.session_state.is_recording:
    time.sleep(2)
    st.rerun()