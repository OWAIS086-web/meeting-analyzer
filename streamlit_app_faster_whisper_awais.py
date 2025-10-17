import warnings
warnings.filterwarnings('ignore')
import os, time, json, pickle, tempfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from audio_processor_faster_whisper import AudioProcessor
from summarizer_2 import MeetingSummarizer

# ================================================================
# FILE STORAGE SETUP
# ================================================================

TEMP_DIR = tempfile.gettempdir()
TRANSCRIPT_FILE = os.path.join(TEMP_DIR, "meeting_transcripts.pkl")
METADATA_FILE = os.path.join(TEMP_DIR, "meeting_metadata.pkl")

def save_transcripts(segments):
    try:
        with open(TRANSCRIPT_FILE, 'wb') as f: pickle.dump(segments, f)
        return True
    except Exception as e:
        print(f"Error saving transcripts: {e}"); return False

def load_transcripts():
    try:
        if os.path.exists(TRANSCRIPT_FILE):
            with open(TRANSCRIPT_FILE, 'rb') as f: return pickle.load(f)
    except Exception as e: print(f"Error loading transcripts: {e}")
    return []

def save_metadata(data):
    try:
        with open(METADATA_FILE, 'wb') as f: pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}"); return False

def load_metadata():
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'rb') as f: return pickle.load(f)
    except Exception as e: print(f"Error loading metadata: {e}")
    return {}

def clear_storage():
    for f in [TRANSCRIPT_FILE, METADATA_FILE]:
        if os.path.exists(f): os.remove(f)

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="AI IT Meeting Analyzer",
    page_icon="💻",
    layout="wide",
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>
:root {
    --primary: #0066CC;
    --secondary: #0099FF;
    --success: #00CC66;
    --warning: #FF9900;
    --danger: #CC0000;
    --bg: #F5F7FA;
}
.stApp {background: var(--bg); font-family: 'Inter', sans-serif;}
.hero {
    background: linear-gradient(135deg, #0066CC, #0099FF);
    color: white; text-align:center;
    border-radius:16px; padding:2.5rem 1.5rem; margin-bottom:2rem;
    box-shadow: 0 4px 12px rgba(0,102,204,0.3);
}
.hero h1 {margin:0; font-size:2.5rem; font-weight:700;}
.hero p {font-size:1.1rem; opacity:0.95; margin-top:0.5rem;}

.card {
    background:white; border-radius:12px;
    padding:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease; margin-bottom:1rem;
}
.card:hover {transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,0.12);}

.feature {
    background: linear-gradient(135deg, #f0f4ff, #e6f0ff);
    border-left: 5px solid var(--secondary);
    border-radius: 10px; padding:1.2rem; height:100%;
}
.feature h4 {color: var(--primary); margin-top:0;}

.recording {
    background:linear-gradient(90deg, var(--danger), #FF3333);
    color:white; text-align:center; padding:1rem;
    border-radius:10px; font-weight:700; font-size:1.1rem;
    animation:pulse 2s infinite; box-shadow: 0 4px 12px rgba(204,0,0,0.4);
}
@keyframes pulse {0%{opacity:1;}50%{opacity:0.85;}100%{opacity:1;}}

.issue-badge {
    background: #FFF3CD; color: #856404; 
    padding: 0.4rem 0.8rem; border-radius: 6px; 
    font-weight: 600; display: inline-block; margin: 0.3rem;
    border-left: 4px solid var(--warning);
}

.recommendation-badge {
    background: #D4EDDA; color: #155724;
    padding: 0.4rem 0.8rem; border-radius: 6px;
    font-weight: 600; display: inline-block; margin: 0.3rem;
    border-left: 4px solid var(--success);
}

.question-badge {
    background: #D1ECF1; color: #0C5460;
    padding: 0.4rem 0.8rem; border-radius: 6px;
    font-weight: 600; display: inline-block; margin: 0.3rem;
    border-left: 4px solid var(--secondary);
}

.footer {
    text-align:center; color:#6B7280; font-size:0.9rem; 
    margin-top:3rem; padding-top:2rem; border-top:1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================

defaults = {
    'is_recording': False,
    'current_analysis': None,
    'final_summary': None,
    'audio_processor': None,
    'summarizer': None,
    'start_time': None,
    'recording_session_id': None
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ================================================================
# CORE CALLBACKS
# ================================================================

def on_new_transcript(text: str):
    segs = load_transcripts(); segs.append(text); save_transcripts(segs)

def start_recording():
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        st.error("❌ Missing XAI_API_KEY! Set environment variable and rerun."); return
    try:
        clear_storage()
        st.session_state.summarizer = MeetingSummarizer(xai_key)
        st.session_state.audio_processor = AudioProcessor(on_new_transcript)
        st.session_state.start_time = datetime.now()
        st.session_state.recording_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_metadata({'start_time': st.session_state.start_time,'session_id': st.session_state.recording_session_id})
        st.session_state.audio_processor.start_recording()
        st.session_state.is_recording = True
        st.success("🎙️ Recording started. Discuss your IT infrastructure, cloud architecture, or technical decisions...")
    except Exception as e:
        st.error(f"Error: {e}")

def stop_recording():
    if st.session_state.audio_processor:
        st.session_state.audio_processor.stop_recording()
    time.sleep(2)
    segments = load_transcripts()
    if len(segments) == 0:
        st.warning("⚠️ No transcripts found! Did you speak during recording?")
        st.session_state.is_recording = False; return
    xai_key = os.getenv("XAI_API_KEY")
    if xai_key:
        with st.spinner("🧠 Generating Final IT Analysis..."):
            fresh = MeetingSummarizer(xai_key)
            for seg in segments: fresh.add_transcript(seg)
            st.session_state.final_summary = fresh.get_final_summary()
    st.session_state.is_recording = False
    st.success("✅ Recording complete! Final IT analysis generated.")

def update_live_analysis():
    if not st.session_state.is_recording or not st.session_state.summarizer: return
    segs = load_transcripts(); current = st.session_state.summarizer.full_transcript
    if len(segs) > len(current):
        for seg in segs[len(current):]:
            analysis = st.session_state.summarizer.add_transcript(seg)
            if analysis: st.session_state.current_analysis = analysis

# ================================================================
# LIVE REFRESH
# ================================================================

if st.session_state.is_recording:
    update_live_analysis()

# ================================================================
# HEADER SECTION
# ================================================================

st.markdown("""
<div class="hero">
    <h1>💻 AI IT Meeting Analyzer</h1>
    <p>Real-time Technical Issue Identification & Best Practice Recommendations</p>
    <p style="font-size:0.9rem; margin-top:0.5rem;">Powered by Grok AI • Cloud • Security • DevOps • Infrastructure</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# FEATURE SHOWCASE
# ================================================================

colA, colB, colC = st.columns(3)
with colA:
    st.markdown("""<div class="feature"><h4>🎤 Real-time Transcription</h4><p>Local inference with Faster-Whisper. Zero cloud dependency.</p></div>""", unsafe_allow_html=True)
with colB:
    st.markdown("""<div class="feature"><h4>🔍 Proactive Issue Detection</h4><p>Identifies security risks, misconfigurations, and architectural pitfalls.</p></div>""", unsafe_allow_html=True)
with colC:
    st.markdown("""<div class="feature"><h4>💡 Smart Recommendations</h4><p>Best practices for Azure, AWS, Kubernetes, and more.</p></div>""", unsafe_allow_html=True)

st.divider()

# ================================================================
# CONTROLS SECTION
# ================================================================

cols = st.columns([1,1,2])
with cols[0]:
    if st.button("▶️ Start Recording", use_container_width=True, disabled=st.session_state.is_recording, type="primary"):
        start_recording()
with cols[1]:
    if st.button("⏹️ Stop Recording", use_container_width=True, disabled=not st.session_state.is_recording):
        stop_recording()
with cols[2]:
    if st.button("🧹 Clear Data", use_container_width=True, disabled=st.session_state.is_recording):
        clear_storage(); st.session_state.final_summary=None; st.session_state.current_analysis=None; st.rerun()

# ================================================================
# RECORDING INDICATOR
# ================================================================

if st.session_state.is_recording:
    dur = (datetime.now() - st.session_state.start_time).seconds
    min_, sec = divmod(dur, 60)
    st.markdown(f"""
    <div class="recording">🔴 LIVE RECORDING - {min_:02d}:{sec:02d} | Listening for IT discussions...</div>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT LAYOUT
# ================================================================

left, right = st.columns([1, 1])
segments = load_transcripts()
full_text = " ".join(segments)

# -------- LEFT: Transcript --------
with left:
    st.markdown("### 📝 Live Transcript")
    if full_text:
        st.markdown(f"""
        <div class="card" style="max-height:500px;overflow-y:auto;">
            <small style="color:#6B7280;">{len(full_text.split())} words • {len(segments)} segments</small>
            <p style="line-height:1.8; color:#111827; margin-top:1rem;">{full_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🎙️ Waiting for IT discussion... Start talking about cloud infrastructure, security, DevOps, etc.")

# -------- RIGHT: IT Insights --------
with right:
    st.markdown("### 💡 Live IT Analysis")
    analysis = st.session_state.final_summary or st.session_state.current_analysis
    
    if analysis:
        # Status badge
        if st.session_state.final_summary:
            st.success("✅ Meeting Complete - Final Analysis")
        else:
            st.info("🔄 Live Analysis - Updates every ~50 words")
        
        # Technical Analysis Overview
        with st.expander("📊 Technical Overview", expanded=True):
            tech_analysis = analysis.get("technical_analysis", "No analysis yet.")
            st.markdown(f"**{tech_analysis}**")
        
        # Potential Issues (Most Important!)
        with st.expander("⚠️ Potential Issues", expanded=True):
            issues = analysis.get("potential_issues", [])
            if issues:
                for i, issue in enumerate(issues, 1):
                    st.markdown(f"""
                    <div class="issue-badge">
                        ⚠️ Issue {i}
                    </div>
                    <p style="margin-left:1rem; margin-top:0.5rem;">{issue}</p>
                    """, unsafe_allow_html=True)
            else:
                st.write("✓ No issues detected in current discussion")
        
        # Recommendations
        with st.expander("✅ Recommendations", expanded=True):
            recs = analysis.get("recommendations", [])
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class="recommendation-badge">
                        ✅ Recommendation {i}
                    </div>
                    <p style="margin-left:1rem; margin-top:0.5rem;">{rec}</p>
                    """, unsafe_allow_html=True)
            else:
                st.write("(No recommendations yet)")
        
        # Clarifying Questions
        with st.expander("❓ Clarifying Questions", expanded=False):
            questions = analysis.get("clarifying_questions", [])
            if questions:
                for i, q in enumerate(questions, 1):
                    st.markdown(f"""
                    <div class="question-badge">
                        ❓ Question {i}
                    </div>
                    <p style="margin-left:1rem; margin-top:0.5rem;">{q}</p>
                    """, unsafe_allow_html=True)
            else:
                st.write("(No questions)")
        
        # Action Items
        with st.expander("📋 Action Items", expanded=False):
            actions = analysis.get("action_items", [])
            if actions:
                for i, a in enumerate(actions, 1):
                    st.markdown(f"**{i}.** {a}")
            else:
                st.write("(No action items identified)")
    else:
        st.info("💬 IT insights will appear after ~50 words of technical discussion.\n\nTry discussing:\n- Cloud infrastructure (AWS, Azure, GCP)\n- Security configurations\n- Kubernetes/container deployments\n- Database architecture\n- Network setup")

# ================================================================
# STATS SECTION
# ================================================================

if segments:
    st.divider()
    st.markdown("### 📊 Meeting Statistics")
    duration = 0
    metadata = load_metadata()
    if metadata.get('start_time'):
        duration = (datetime.now() - metadata['start_time']).seconds // 60
    colx = st.columns(4)
    colx[0].metric("Duration", f"{duration} min")
    colx[1].metric("Words", len(full_text.split()))
    colx[2].metric("Segments", len(segments))
    colx[3].metric("Characters", len(full_text))
    
    # Show issue/recommendation counts if available
    if st.session_state.summarizer:
        stats = st.session_state.summarizer.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Issues Identified", stats.get('issues_identified', 0))
        with col2:
            st.metric("Recommendations", stats.get('recommendations_given', 0))
    
    with st.expander("🔧 Debug Info"):
        st.code(f"Storage: {TRANSCRIPT_FILE}\nSegments: {len(segments)}\nSession: {st.session_state.recording_session_id}")

# ================================================================
# FOOTER
# ================================================================

st.markdown("""
<div class="footer">
<p><strong>🔒 Privacy First:</strong> Local transcription with Faster-Whisper • Only analysis sent to Grok API</p>
<p><strong>💰 Cost Efficient:</strong> ~$0.02 per 10-minute meeting (transcription free, Grok analysis $0.02)</p>
<p><strong>🎯 IT Domains:</strong> Cloud (Azure/AWS/GCP) • Kubernetes • Security • DevOps • Networking • Database Architecture</p>
<p style="margin-top:1rem; font-size:0.85rem;">Developed for IT teams seeking real-time technical guidance during meetings</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# AUTO REFRESH FOR LIVE UPDATES
# ================================================================

if st.session_state.is_recording:
    time.sleep(2)
    st.rerun()