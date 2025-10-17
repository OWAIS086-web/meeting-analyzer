"""
Terminal-Based Meeting Analyzer
No Streamlit - just pure Python in terminal
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["ORT_DISABLE_DEVICE_DISCOVERY"] = "1"

import warnings
warnings.filterwarnings('ignore')

from audio_processor_faster_whisper import AudioProcessor
from summarizer import MeetingSummarizer
import time
from datetime import datetime
import sys

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

class MeetingAnalyzerTerminal:
    def __init__(self, api_key):
        self.api_key = api_key
        self.transcripts = []
        self.full_text = ""
        self.summarizer = None
        self.audio_processor = None
        self.start_time = None
        
    def on_transcript(self, text):
        """Callback when new transcript arrives"""
        self.transcripts.append(text)
        self.full_text += text + " "
        
        print(f"\n{Colors.GREEN}✓ Transcribed:{Colors.END} {text}")
        
        # Add to summarizer
        if self.summarizer:
            analysis = self.summarizer.add_transcript(text)
            
            # Show analysis every 50 words
            if analysis:
                self.print_analysis(analysis, is_final=False)
    
    def print_analysis(self, analysis, is_final=False):
        """Print analysis in formatted way"""
        if is_final:
            print(f"\n{'='*70}")
            print(f"{Colors.BOLD}{Colors.BLUE}📋 FINAL MEETING SUMMARY{Colors.END}")
            print(f"{'='*70}\n")
        else:
            print(f"\n{Colors.YELLOW}{'─'*70}")
            print(f"💡 Live Insights (interim)")
            print(f"{'─'*70}{Colors.END}\n")
        
        # Summary
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  {analysis.get('summary', 'No summary')}\n")
        
        # Action Items
        print(f"{Colors.BOLD}Action Items:{Colors.END}")
        items = analysis.get('action_items', [])
        if items:
            for item in items:
                print(f"  ✓ {item}")
        else:
            print(f"  (none yet)")
        print()
        
        # IT Insights
        print(f"{Colors.BOLD}IT Insights:{Colors.END}")
        insights = analysis.get('it_insights', [])
        if insights:
            for insight in insights:
                print(f"  💻 {insight}")
        else:
            print(f"  (none yet)")
        print()
        
        # Key Decisions
        print(f"{Colors.BOLD}Key Decisions:{Colors.END}")
        decisions = analysis.get('key_decisions', [])
        if decisions:
            for decision in decisions:
                print(f"  🎯 {decision}")
        else:
            print(f"  (none yet)")
        
        if not is_final:
            print(f"\n{Colors.YELLOW}{'─'*70}{Colors.END}\n")
    
    def start_recording(self, duration_seconds=None):
        """Start recording meeting"""
        print(f"\n{Colors.HEADER}{'='*70}")
        print("🎤 AI MEETING ANALYZER - Terminal Mode")
        print(f"{'='*70}{Colors.END}\n")
        
        # Initialize components
        print(f"{Colors.BLUE}Initializing...{Colors.END}")
        self.summarizer = MeetingSummarizer(self.api_key)
        self.audio_processor = AudioProcessor(self.on_transcript)
        self.transcripts = []
        self.full_text = ""
        self.start_time = datetime.now()
        
        # Start recording
        self.audio_processor.start_recording()
        
        print(f"\n{Colors.GREEN}✓ Recording started!{Colors.END}")
        print(f"{Colors.BOLD}🎤 Speak into your microphone...{Colors.END}")
        
        if duration_seconds:
            print(f"Recording for {duration_seconds} seconds...")
            print(f"(Press Ctrl+C to stop early)\n")
        else:
            print(f"Press Ctrl+C when done\n")
        
        try:
            if duration_seconds:
                # Countdown with updates
                for remaining in range(duration_seconds, 0, -10):
                    time.sleep(min(10, remaining))
                    elapsed = duration_seconds - remaining + 10
                    print(f"{Colors.BLUE}⏱️  {elapsed}s elapsed... (transcript segments: {len(self.transcripts)}){Colors.END}")
            else:
                # Wait indefinitely
                while True:
                    time.sleep(10)
                    elapsed = (datetime.now() - self.start_time).seconds
                    print(f"{Colors.BLUE}⏱️  {elapsed}s elapsed... (transcript segments: {len(self.transcripts)}){Colors.END}")
                    
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⏹️  Stopping recording...{Colors.END}")
        
        # Stop recording
        self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and generate final summary"""
        if self.audio_processor:
            self.audio_processor.stop_recording()
        
        # Wait for final processing
        time.sleep(1)
        
        # Show statistics
        duration = (datetime.now() - self.start_time).seconds
        word_count = len(self.full_text.split())
        
        print(f"\n{Colors.HEADER}{'='*70}")
        print("📊 RECORDING STATISTICS")
        print(f"{'='*70}{Colors.END}")
        print(f"  Duration: {duration // 60}m {duration % 60}s")
        print(f"  Transcript segments: {len(self.transcripts)}")
        print(f"  Total words: {word_count}")
        print(f"  Characters: {len(self.full_text)}")
        
        if len(self.transcripts) == 0:
            print(f"\n{Colors.RED}❌ No transcripts recorded!{Colors.END}")
            print("Make sure you spoke during the recording.")
            return
        
        # Generate final summary
        print(f"\n{Colors.BLUE}Generating final comprehensive summary...{Colors.END}")
        final_summary = self.summarizer.get_final_summary()
        
        self.print_analysis(final_summary, is_final=True)
        
        # Show full transcript
        print(f"\n{Colors.HEADER}{'='*70}")
        print("📝 FULL TRANSCRIPT")
        print(f"{'='*70}{Colors.END}")
        print(f"\n{self.full_text}\n")
        
        print(f"{Colors.GREEN}✅ Meeting analysis complete!{Colors.END}\n")


def main():
    """Main function"""
    # Get API key
    api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        print(f"{Colors.RED}❌ Error: XAI_API_KEY environment variable not set{Colors.END}")
        print("\nSet it with:")
        print("  Windows: $env:XAI_API_KEY='xai-your-key'")
        print("  Linux/Mac: export XAI_API_KEY='xai-your-key'")
        sys.exit(1)
    
    # Create analyzer
    analyzer = MeetingAnalyzerTerminal(api_key)
    
    # Ask for duration
    print(f"\n{Colors.BOLD}Recording Duration:{Colors.END}")
    print("1. Quick test (30 seconds)")
    print("2. Short meeting (2 minutes)")
    print("3. Medium meeting (5 minutes)")
    print("4. Long meeting (10 minutes)")
    print("5. Custom duration")
    print("6. Manual stop (press Ctrl+C when done)")
    
    choice = input(f"\n{Colors.BLUE}Select option (1-6):{Colors.END} ").strip()
    
    duration_map = {
        '1': 30,
        '2': 120,
        '3': 300,
        '4': 600,
        '5': None,
        '6': None
    }
    
    duration = duration_map.get(choice)
    
    if choice == '5':
        try:
            duration = int(input(f"{Colors.BLUE}Enter duration in seconds:{Colors.END} "))
        except:
            print(f"{Colors.RED}Invalid input, using 60 seconds{Colors.END}")
            duration = 60
    
    # Start recording
    analyzer.start_recording(duration)


if __name__ == "__main__":
    main()