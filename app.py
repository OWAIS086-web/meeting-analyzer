
import warnings
warnings.filterwarnings('ignore')

import os
import time
import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
from audio_processor_faster_whisper import AudioProcessor
from summarizer_2 import MeetingSummarizer
import threading

# Load environment variables from .env file
load_dotenv()

# ================================================================
# FLASK APPLICATION SETUP
# ================================================================

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global application state
app_state = {
    'is_recording': False,
    'current_analysis': None,
    'final_summary': None,
    'audio_processor': None,
    'summarizer': None,
    'start_time': None,
    'recording_session_id': None,
    'total_sessions': 0,
    'total_meetings_analyzed': 0
}

# ================================================================
# FILE STORAGE CONFIGURATION
# ================================================================

TEMP_DIR = tempfile.gettempdir()
TRANSCRIPT_FILE = os.path.join(TEMP_DIR, "meeting_transcripts.pkl")
METADATA_FILE = os.path.join(TEMP_DIR, "meeting_metadata.pkl")
STATS_FILE = os.path.join(TEMP_DIR, "app_stats.pkl")

# ================================================================
# DATA PERSISTENCE FUNCTIONS
# ================================================================

def save_transcripts(segments):
    """Save transcript segments to persistent storage."""
    try:
        with open(TRANSCRIPT_FILE, 'wb') as f:
            pickle.dump(segments, f)
        return True
    except Exception as e:
        print(f"Error saving transcripts: {e}")
        return False

def load_transcripts():
    """Load transcript segments from persistent storage."""
    try:
        if os.path.exists(TRANSCRIPT_FILE):
            with open(TRANSCRIPT_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading transcripts: {e}")
    return []

def save_metadata(data):
    """Save session metadata to persistent storage."""
    try:
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def load_metadata():
    """Load session metadata from persistent storage."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
    return {}

def save_app_stats():
    """Save application statistics."""
    try:
        stats = {
            'total_sessions': app_state['total_sessions'],
            'total_meetings_analyzed': app_state['total_meetings_analyzed'],
            'last_updated': datetime.now()
        }
        with open(STATS_FILE, 'wb') as f:
            pickle.dump(stats, f)
    except Exception as e:
        print(f"Error saving app stats: {e}")

def load_app_stats():
    """Load application statistics."""
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'rb') as f:
                stats = pickle.load(f)
                app_state['total_sessions'] = stats.get('total_sessions', 0)
                app_state['total_meetings_analyzed'] = stats.get('total_meetings_analyzed', 0)
    except Exception as e:
        print(f"Error loading app stats: {e}")

def clear_storage():
    """Clear all persistent storage files."""
    for f in [TRANSCRIPT_FILE, METADATA_FILE]:
        if os.path.exists(f):
            os.remove(f)

# ================================================================
# CORE BUSINESS LOGIC
# ================================================================

def on_new_transcript(text: str):
    """Callback function for new transcript segments."""
    try:
        segments = load_transcripts()
        segments.append(text)
        save_transcripts(segments)
        print(f"New transcript segment: {text[:50]}...")
        
        # Trigger live analysis update
        if app_state['is_recording'] and app_state['summarizer']:
            try:
                analysis = app_state['summarizer'].add_transcript(text)
                if analysis:
                    app_state['current_analysis'] = analysis
                    print("Live analysis updated")
            except Exception as e:
                print(f"Error updating live analysis: {e}")
                
    except Exception as e:
        print(f"Error in transcript callback: {e}")
        import traceback
        traceback.print_exc()

def start_recording():
    """Start the recording session with audio processing and AI analysis."""
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        return {
            "success": False, 
            "message": "❌ Missing XAI_API_KEY! Please set your API key in the .env file."
        }
    
    try:
        # Clear previous session data
        clear_storage()
        
        # Initialize AI components
        print("Initializing AI components...")
        app_state['summarizer'] = MeetingSummarizer(xai_key)
        
        print("Initializing audio processor...")
        app_state['audio_processor'] = AudioProcessor(on_new_transcript)
        
        # Set session metadata
        app_state['start_time'] = datetime.now()
        app_state['recording_session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save session metadata
        metadata = {
            'start_time': app_state['start_time'],
            'session_id': app_state['recording_session_id'],
            'xai_model': 'grok-4-fast-reasoning'
        }
        save_metadata(metadata)
        
        # Start audio processing
        print("Starting audio recording...")
        app_state['audio_processor'].start_recording()
        
        # Check if recording actually started
        if not app_state['audio_processor'].is_recording:
            return {
                "success": False,
                "message": "❌ Failed to start audio recording. Please check your microphone permissions and ensure no other app is using the microphone."
            }
        
        app_state['is_recording'] = True
        
        # Update statistics
        app_state['total_sessions'] += 1
        save_app_stats()
        
        print("Recording started successfully!")
        return {
            "success": True, 
            "message": "🎙️ Recording started successfully! Speak clearly into your microphone to begin transcription."
        }
        
    except Exception as e:
        print(f"Error starting recording: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False, 
            "message": f"❌ Error starting recording: {str(e)}. Please check the console for more details."
        }

def stop_recording():
    """Stop the recording session and generate final analysis."""
    try:
        # Stop audio processing
        if app_state['audio_processor']:
            app_state['audio_processor'].stop_recording()
        
        # Allow time for final processing
        time.sleep(2)
        
        # Get all transcript segments
        segments = load_transcripts()
        
        if len(segments) == 0:
            app_state['is_recording'] = False
            return {
                "success": False, 
                "message": "⚠️ No transcripts found! Please ensure your microphone is working and you spoke during the recording."
            }
        
        # Generate final analysis
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            fresh_summarizer = MeetingSummarizer(xai_key)
            for segment in segments:
                fresh_summarizer.add_transcript(segment)
            app_state['final_summary'] = fresh_summarizer.get_final_summary()
        
        # Update session state
        app_state['is_recording'] = False
        app_state['total_meetings_analyzed'] += 1
        save_app_stats()
        
        return {
            "success": True, 
            "message": "✅ Recording completed successfully! Final IT analysis has been generated."
        }
        
    except Exception as e:
        app_state['is_recording'] = False
        return {
            "success": False, 
            "message": f"❌ Error stopping recording: {str(e)}"
        }

def update_live_analysis():
    """Update live analysis during recording."""
    if not app_state['is_recording'] or not app_state['summarizer']:
        return
    
    try:
        segments = load_transcripts()
        current_transcript = app_state['summarizer'].full_transcript
        
        # Process new segments
        if len(segments) > len(current_transcript):
            for segment in segments[len(current_transcript):]:
                analysis = app_state['summarizer'].add_transcript(segment)
                if analysis:
                    app_state['current_analysis'] = analysis
                    
    except Exception as e:
        print(f"Error updating live analysis: {e}")

def calculate_metrics():
    """Calculate real-time metrics for the dashboard."""
    segments = load_transcripts()
    full_text = " ".join(segments)
    metadata = load_metadata()
    
    # Calculate duration
    duration = 0
    if metadata.get('start_time'):
        duration = (datetime.now() - metadata['start_time']).seconds
    
    # Calculate basic metrics
    word_count = len(full_text.split()) if full_text else 0
    wpm = round(word_count / (duration / 60)) if duration > 0 else 0
    
    # Calculate confidence (mock implementation - could be enhanced with actual confidence scores)
    confidence = min(95, 75 + (word_count // 10)) if word_count > 0 else 0
    
    # Calculate momentum (discussion flow)
    momentum = min(100, word_count // 10) if word_count > 0 else 0
    
    # Calculate technical depth
    tech_keywords = ['cloud', 'aws', 'azure', 'kubernetes', 'docker', 'security', 'api', 'database', 
                    'server', 'network', 'infrastructure', 'devops', 'ci/cd', 'deployment']
    tech_mentions = sum(1 for word in full_text.lower().split() if word in tech_keywords)
    tech_depth = min(100, tech_mentions * 5) if full_text else 0
    
    # Calculate insights count
    insights = 0
    if app_state['current_analysis']:
        insights += len(app_state['current_analysis'].get('potential_issues', []))
        insights += len(app_state['current_analysis'].get('recommendations', []))
    
    return {
        'duration': duration,
        'word_count': word_count,
        'wpm': wpm,
        'confidence': confidence,
        'momentum': momentum,
        'tech_depth': tech_depth,
        'insights': insights,
        'segments_count': len(segments),
        'full_text': full_text
    }

# ================================================================
# FLASK ROUTES
# ================================================================

@app.route('/')
def index():
    """Main application dashboard."""
    # Update live analysis if recording
    if app_state['is_recording']:
        update_live_analysis()
    
    # Calculate current metrics
    metrics = calculate_metrics()
    
    # Determine status
    status_text = "READY TO RECORD"
    status_color = "#10B981"
    if app_state['is_recording']:
        status_text = "RECORDING IN PROGRESS"
        status_color = "#EF4444"
    
    # Prepare template data
    template_data = {
        'is_recording': app_state['is_recording'],
        'status_text': status_text,
        'status_color': status_color,
        'analysis': app_state['final_summary'] or app_state['current_analysis'],
        'final_summary': app_state['final_summary'],
        'session_id': app_state['recording_session_id'],
        'total_sessions': app_state['total_sessions'],
        'total_meetings_analyzed': app_state['total_meetings_analyzed'],
        **metrics
    }
    
    return render_template('index.html', **template_data)

@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    """API endpoint to start recording."""
    result = start_recording()
    return jsonify(result)

@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    """API endpoint to stop recording."""
    result = stop_recording()
    return jsonify(result)

@app.route('/clear_data', methods=['POST'])
def clear_data_route():
    """API endpoint to clear all session data."""
    if not app_state['is_recording']:
        clear_storage()
        app_state['final_summary'] = None
        app_state['current_analysis'] = None
        return jsonify({
            "success": True, 
            "message": "🗑️ All data cleared successfully!"
        })
    return jsonify({
        "success": False, 
        "message": "❌ Cannot clear data while recording is in progress."
    })

@app.route('/api/status')
def get_status():
    """API endpoint to get current status and metrics."""
    metrics = calculate_metrics()
    
    return jsonify({
        'is_recording': app_state['is_recording'],
        'session_id': app_state['recording_session_id'],
        'analysis': app_state['final_summary'] or app_state['current_analysis'],
        'final_summary': bool(app_state['final_summary']),
        'total_sessions': app_state['total_sessions'],
        'total_meetings_analyzed': app_state['total_meetings_analyzed'],
        **metrics
    })

@app.route('/api/transcript')
def get_transcript():
    """API endpoint to get current transcript."""
    segments = load_transcripts()
    return jsonify({
        'segments': segments,
        'full_text': " ".join(segments),
        'segment_count': len(segments)
    })

@app.route('/export/summary')
def export_summary():
    """Export final summary as JSON."""
    if app_state['final_summary']:
        return jsonify({
            'session_id': app_state['recording_session_id'],
            'timestamp': datetime.now().isoformat(),
            'summary': app_state['final_summary'],
            'transcript': " ".join(load_transcripts())
        })
    return jsonify({'error': 'No summary available'}), 404

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500

# ================================================================
# APPLICATION INITIALIZATION
# ================================================================

def initialize_app():
    """Initialize the application on startup."""
    print("🚀 Initializing AI IT Meeting Analyzer...")
    
    # Load application statistics
    load_app_stats()
    
    # Check for required API key
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        print("⚠️  WARNING: XAI_API_KEY not found in environment variables!")
        print("   Please set your xAI API key in the .env file.")
        print("   Get your API key from: https://console.x.ai")
    else:
        print("✅ XAI API key loaded successfully")
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"📁 Created templates directory: {templates_dir}")
    
    print(f"📊 Total sessions: {app_state['total_sessions']}")
    print(f"📈 Total meetings analyzed: {app_state['total_meetings_analyzed']}")
    print("🎙️  AI IT Meeting Analyzer is ready!")
    print("🌐 Open http://localhost:5000 in your browser to start")

# ================================================================
# MAIN APPLICATION ENTRY POINT
# ================================================================

if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    # Run the Flask development server
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )