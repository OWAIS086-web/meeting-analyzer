# AI IT Meeting Analyzer

Real-time meeting transcription and analysis using Faster-Whisper (local) and Grok AI.

## Features

- Real-time local transcription with Faster-Whisper (no cloud costs)
- Voice Activity Detection (VAD) for accurate speech segmentation
- IT-focused analysis with Grok AI
- Proactive technical issue identification
- Best practice recommendations for cloud infrastructure, security, and DevOps
- Progressive summarization for long meetings
- Persistent storage across sessions

## System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB recommended for larger Whisper models)
- Microphone for audio input
- Internet connection for Grok API

## Installation

### Step 1: Install System Dependencies

**macOS:**
```bash
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows:**
```bash
# PyAudio will be installed via pip
# No additional system packages required
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a .env file in the project root directory:

```bash
XAI_API_KEY=xai-your-api-key-here
```

To get your xAI API key:
1. Go to https://console.x.ai
2. Sign up or log in
3. Click "Create API Key"
4. Copy the key (starts with xai-)
5. Add it to your .env file

## API Costs

### Transcription (Faster-Whisper)
- Cost: Free (runs locally on your machine)
- No internet required for transcription
- Privacy-focused: audio never leaves your device

### Analysis (Grok API)
- Model: grok-4-fast-reasoning
- Cost: Approximately $0.0002 per 1,000 tokens
- Average 10-minute meeting: 500-1,000 tokens
- Estimated cost per meeting: $0.001 - $0.002 (less than 1 cent)
- Free credits: $25 (sufficient for 10,000+ meetings)

### Total Cost Per Meeting
- Transcription: $0.00 (local)
- Analysis: ~$0.002 (Grok)
- Total: ~$0.002 per 10-minute meeting

## Usage

### main application  Web Interface (high Recommended)

```bash
app.py
open http://127.0.0.1:5000/ 
```

This will open a web interface in your browser with:
- Live transcription display
- Real-time IT analysis from Grok
- Issue detection and recommendations
- Session statistics and debug info
- 
### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app_faster_whisper_awais.py
```

This will open a web interface in your browser with:
- Live transcription display
- Real-time IT analysis from Grok
- Issue detection and recommendations
- Session statistics and debug info

### Option 2: Terminal Interface

```bash
python terminal_meeting_analyzer.py
```

This provides a command-line interface with:
- No browser required
- Colored terminal output
- Configurable recording duration
- Final summary export

### Option 3: Test Individual Components

Test audio transcription only:
```bash
python audio_processor_faster_whisper.py
```

Test Grok analysis only:
```bash
python summarizer_2.py
```

## Configuration

### Audio Settings

Edit audio_processor_faster_whisper.py to adjust:

```python
CHUNK_DURATION_SECONDS = 3     # Process audio every 3 seconds
OVERLAP_SECONDS = 1.5          # 1.5-second overlap for continuity
WHISPER_MODEL_SIZE = "base"    # Options: tiny, base, small, medium, large
DEVICE = "cpu"                 # Use "cuda" for GPU acceleration
```

### Analysis Settings

Edit summarizer_2.py to adjust:

```python
WORDS_PER_ANALYSIS = 50        # Analyze every 50 words
WORDS_PER_ROLLING_SUMMARY = 300  # Rolling summary every 300 words
MAX_PRIOR_SUMMARY_WORDS = 1000   # Context window size
GROK_MODEL = "grok-4-fast-reasoning"  # Grok model selection
```

## What the System Does

### 1. Real-time Transcription
- Captures audio from your microphone
- Transcribes speech using Faster-Whisper locally
- Processes audio in 3-second chunks with overlap for accuracy
- No cloud dependency for transcription

### 2. IT-Focused Analysis
Grok AI analyzes the discussion and provides:

- Technical Overview: Summary of what is being discussed
- Potential Issues: Security risks, misconfigurations, architectural problems
- Recommendations: Best practices and specific solutions
- Clarifying Questions: Probes for missing or ambiguous details
- Action Items: Tasks with owners when mentioned

### 3. Progressive Summarization
For long meetings:
- Creates rolling summaries every 300 words
- Maintains context without overwhelming the API
- Generates comprehensive final summary when recording stops

## IT Domains Covered

- Cloud Services (Azure, AWS, GCP)
- Kubernetes and Container Orchestration
- Infrastructure and Networking
- DevOps and CI/CD Pipelines
- Cybersecurity and Compliance
- Software Architecture
- Database Design
- System Administration

## Example Use Cases

### Scenario 1: Azure AKS Deployment
Discussion: "We're deploying an AKS cluster with the jump box on a separate subnet connected via peering."

Grok Analysis:
- Issue: Jump box on different subnet can cause connectivity problems
- Recommendation: Deploy in same VNet, use NSGs for isolation
- Question: Is the jump box intended as a bastion host?

### Scenario 2: AWS Security Configuration
Discussion: "Let's open port 22 from any IP to make SSH testing easier."

Grok Analysis:
- Issue: Opening SSH to 0.0.0.0/0 exposes instances to brute-force attacks
- Recommendation: Restrict to specific IP ranges, consider AWS SSM Session Manager
- Question: What is the expected source IP range for connections?

## Troubleshooting

### Microphone Issues

If you see "Error opening microphone":
1. Check that no other application is using your microphone
2. Grant microphone permissions to your terminal/IDE
3. Try listing available devices:
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))
```

### No Transcripts Appearing

1. Verify you are speaking clearly into the microphone
2. Check that audio levels are adequate (not too quiet)
3. Try running the audio processor test:
```bash
python audio_processor_faster_whisper.py
```

### API Connection Issues

1. Verify XAI_API_KEY is set correctly in .env
2. Check internet connection
3. Verify API key is active at https://console.x.ai

### Performance Issues

If transcription is slow:
1. Use a smaller Whisper model (tiny or base)
2. Reduce CHUNK_DURATION_SECONDS
3. Enable GPU acceleration if available (set DEVICE = "cuda")

## File Structure

```
project/
├── audio_processor_faster_whisper.py  # Audio capture and transcription
├── summarizer_2.py                      # Grok API integration and analysis
├── streamlit_app_faster_whisper_awais.py  # Web UI
├── terminal_meeting_analyzer.py       # Terminal UI
├── requirements.txt                   # Python dependencies
├── .env                              # API keys (create this)
└── README.md                         # This file
```

## Requirements.txt

```
streamlit>=1.28.0
pyaudio>=0.2.13
numpy>=1.24.0
faster-whisper>=0.10.0
openai>=1.3.0
python-dotenv>=1.0.0
```

## Privacy and Security

- Transcription happens entirely on your local machine
- Audio data never leaves your device
- Only transcript text is sent to Grok API for analysis
- No audio recordings are stored unless explicitly saved
- API keys stored in .env file (add to .gitignore)

## Contributing

This is a custom system built for IT meeting analysis. To modify:
1. Update SYSTEM_PROMPT in summarizer_2.py for different analysis focus
2. Adjust audio settings in audio_processor_faster_whisper.py
3. Customize UI in streamlit_app_faster_whisper_awais.py

## License

Custom internal tool for IT team use.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies are installed correctly
3. Test individual components (audio processor, summarizer_2) separately

4. Check Grok API status at https://status.x.ai

