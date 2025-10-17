"""
Summarizer Module
Handles interaction with Grok API for meeting analysis.

Key Components:
1. Grok API integration (xAI)
2. Rolling summarization for long meetings
3. Context management and progressive summarization
"""

import os
from openai import OpenAI
from typing import Dict, List, Optional
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - Adjust these based on your needs
# ============================================================================

# Summarization triggers
WORDS_PER_ANALYSIS = 50        # Analyze every 50 words (real-time insights)
WORDS_PER_ROLLING_SUMMARY = 300  # Create rolling summary every ~2 min
                                # (150 words/min * 2 min = 300 words)

# Context management
MAX_PRIOR_SUMMARY_WORDS = 1000  # Maximum words to keep in rolling context
                                # ~30% of Grok's context window
                                # Increase this for longer meetings

# Grok model selection
GROK_MODEL = "grok-4-fast-reasoning"  # Options:
                                      # - grok-4-fast-reasoning (RECOMMENDED, cheapest)
                                      # - grok-4 (more powerful, expensive)
                                      # - grok-3 (legacy)

# ============================================================================

# System prompt for Grok
# This defines how Grok should analyze the meeting
SYSTEM_PROMPT = """
You are a meeting analysis AI assistant. Analyze the provided meeting transcript incrementally and provide actionable insights.

Output MUST be valid JSON with these exact keys:
{
  "summary": "A concise summary of the discussion so far (2-3 sentences)",
  "action_items": ["List of action items with owners if mentioned", "Example: Bob to draft report by Friday"],
  "it_insights": ["Domain-specific commentary, especially IT-related", "Example: Consider using REST API for the integration"],
  "key_decisions": ["Any decisions made", "Example: Decided to proceed with Option A"]
}

Guidelines:
- For ongoing analysis: Build on prior context and focus on NEW information
- For final summary: Provide comprehensive overview of entire meeting
- Be specific and actionable
- Extract names and deadlines when mentioned
- Identify technical terms and suggestions
- Keep summaries concise but informative
"""


class MeetingSummarizer:
    """
    Handles meeting analysis using Grok API.
    
    Architecture:
    1. Accumulates transcript text
    2. Analyzes every N words (real-time insights)
    3. Creates rolling summaries every M words (context compression)
    4. Maintains context for long meetings using progressive summarization
    """
    
    def __init__(self, api_key_grok: str):
        """
        Initialize Grok API client.
        
        Args:
            api_key: xAI API key from https://console.x.ai
        
        How Grok API works:
        - Base URL: https://api.x.ai/v1
        - Compatible with OpenAI SDK (same interface)
        - Input: Messages (system prompt + user content)
        - Output: JSON response from model
        """
        # Initialize OpenAI-compatible client for Grok
        # We use OpenAI's SDK because Grok API is compatible
        self.client = OpenAI(
            api_key=api_key_grok,
            base_url="https://api.x.ai/v1"  # Grok API endpoint
        )
        
        # Transcript accumulation
        self.full_transcript: List[str] = []  # All transcript segments
        self.word_count = 0                   # Total words transcribed
        self.last_analysis_word_count = 0     # Words at last analysis
        
        # Rolling summaries for context management
        self.rolling_summaries: List[str] = []  # Progressive summaries
        self.last_summary_word_count = 0        # Words at last summary
        
        # Latest analysis results
        self.latest_analysis: Optional[Dict] = None
        
        # Meeting metadata
        self.start_time = datetime.now()
    
    def add_transcript(self, text: str) -> Optional[Dict]:
        """
        Add new transcript text and analyze if threshold reached.
        
        Args:
            text: New transcript segment from audio processor
        
        Returns:
            Analysis dict if threshold reached, None otherwise
        
        How it works:
        1. Add text to buffer
        2. Count words
        3. If >= WORDS_PER_ANALYSIS, analyze and return results
        """
        # Add to transcript
        self.full_transcript.append(text)
        words = text.split()
        self.word_count += len(words)
        
        # Check if we should analyze
        words_since_last_analysis = self.word_count - self.last_analysis_word_count
        
        if words_since_last_analysis >= WORDS_PER_ANALYSIS:
            print(f"📊 Analyzing ({self.word_count} total words)...")
            analysis = self._analyze_current_state()
            self.last_analysis_word_count = self.word_count
            return analysis
        
        return None
    
    def _analyze_current_state(self) -> Dict:
        """
        Analyze current meeting state using Grok API.
        
        How Progressive Summarization works:
        1. Check if we need a rolling summary (every ~300 words)
        2. Build context from prior summaries + recent transcript
        3. Send to Grok for analysis
        4. Update rolling summaries if needed
        
        Returns:
            Dict with keys: summary, action_items, it_insights, key_decisions
        """
        # Step 1: Check if we need a new rolling summary
        words_since_last_summary = self.word_count - self.last_summary_word_count
        
        if words_since_last_summary >= WORDS_PER_ROLLING_SUMMARY:
            print("📝 Creating rolling summary...")
            self._create_rolling_summary()
            self.last_summary_word_count = self.word_count
        
        # Step 2: Build context for analysis
        context = self._build_context()
        
        # Step 3: Call Grok API
        try:
            # Make API call to Grok
            # This is where the magic happens - sending data to xAI's servers
            response = self.client.chat.completions.create(
                model=GROK_MODEL,
                
                # Messages format: [system, user]
                # - system: Instructions for the AI
                # - user: The content to analyze
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                
                # Response settings
                temperature=0.3,  # Lower = more consistent, Higher = more creative
                max_tokens=1000,  # Maximum words in response
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Extract response
            # response.choices[0] = first (and only) response
            # .message.content = the actual text from Grok
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            analysis = json.loads(response_text)
            
            # Validate response has required keys
            required_keys = ["summary", "action_items", "it_insights", "key_decisions"]
            for key in required_keys:
                if key not in analysis:
                    analysis[key] = [] if key != "summary" else "No summary available"
            
            # Store latest analysis
            self.latest_analysis = analysis
            
            # Print token usage for cost tracking
            # Grok charges per token (input + output)
            if hasattr(response, 'usage'):
                print(f"💰 Tokens used: {response.usage.total_tokens} "
                      f"(input: {response.usage.prompt_tokens}, "
                      f"output: {response.usage.completion_tokens})")
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing JSON from Grok: {e}")
            print(f"Response was: {response_text}")
            return self._get_error_response("Invalid JSON response from API")
        
        except Exception as e:
            print(f"❌ Error calling Grok API: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_response(str(e))
    
    def _build_context(self) -> str:
        """
        Build context string for Grok analysis.
        
        Context Structure:
        [Prior Summaries] + [Recent Transcript] + [Current Segment]
        
        This ensures Grok has:
        1. Historical context (compressed via summaries)
        2. Recent detailed transcript
        3. Current segment for analysis
        """
        context_parts = []
        
        # Add meeting metadata
        duration = (datetime.now() - self.start_time).seconds // 60
        context_parts.append(
            f"MEETING METADATA:\n"
            f"- Duration: {duration} minutes\n"
            f"- Total words: {self.word_count}\n"
        )
        
        # Add rolling summaries (compressed historical context)
        if self.rolling_summaries:
            context_parts.append("\n--- PREVIOUS DISCUSSION (SUMMARIES) ---")
            for i, summary in enumerate(self.rolling_summaries):
                context_parts.append(f"\nSummary {i+1}:\n{summary}")
        
        # Add recent transcript (last ~500 words or last 3 segments)
        recent_segments = self.full_transcript[-3:]  # Last 3 segments
        if recent_segments:
            context_parts.append("\n--- RECENT TRANSCRIPT ---")
            context_parts.append(" ".join(recent_segments))
        
        return "\n".join(context_parts)
    
    def _create_rolling_summary(self):
        """
        Create a compressed summary of recent discussion.
        
        This is the heart of handling long meetings:
        1. Take recent transcript segments
        2. Summarize them concisely
        3. Add to rolling summaries
        4. Trim old summaries if needed
        
        This prevents context from growing infinitely while
        retaining key information.
        """
        # Get segments since last summary
        recent_text = " ".join(self.full_transcript[-5:])  # Last 5 segments
        
        try:
            # Create concise summary
            response = self.client.chat.completions.create(
                model=GROK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this meeting segment concisely in 2-3 sentences. Focus on key points, decisions, and action items."
                    },
                    {
                        "role": "user",
                        "content": recent_text
                    }
                ],
                temperature=0.3,
                max_tokens=200  # Keep summary short
            )
            
            summary = response.choices[0].message.content
            self.rolling_summaries.append(summary)
            
            # Trim old summaries if context too long
            total_summary_words = sum(
                len(s.split()) for s in self.rolling_summaries
            )
            
            while total_summary_words > MAX_PRIOR_SUMMARY_WORDS:
                # Remove oldest summary
                removed = self.rolling_summaries.pop(0)
                total_summary_words -= len(removed.split())
                print(f"🗑️ Trimmed old summary to maintain context size")
            
            print(f"✓ Rolling summary created ({len(self.rolling_summaries)} total)")
            
        except Exception as e:
            print(f"⚠️ Error creating rolling summary: {e}")
    
    def get_final_summary(self) -> Dict:
        """
        Generate comprehensive final summary of entire meeting.
        
        Called when user clicks "Stop Recording".
        Uses all rolling summaries + full transcript.
        
        Returns:
            Comprehensive analysis dict
        """
        print("📋 Generating final comprehensive summary...")
        
        # Build comprehensive context
        context_parts = [
            f"MEETING COMPLETED\n"
            f"Duration: {(datetime.now() - self.start_time).seconds // 60} minutes\n"
            f"Total words: {self.word_count}\n"
        ]
        
        # Add all summaries
        if self.rolling_summaries:
            context_parts.append("\n--- MEETING PROGRESSION (SUMMARIES) ---")
            for i, summary in enumerate(self.rolling_summaries):
                context_parts.append(f"\nPhase {i+1}:\n{summary}")
        
        # Add full transcript (or last ~1000 words if too long)
        full_text = " ".join(self.full_transcript)
        words = full_text.split()
        if len(words) > 1000:
            # Use last 1000 words only
            full_text = " ".join(words[-1000:])
            context_parts.append("\n--- RECENT TRANSCRIPT (LAST 1000 WORDS) ---")
        else:
            context_parts.append("\n--- FULL TRANSCRIPT ---")
        
        context_parts.append(full_text)
        
        context = "\n".join(context_parts)
        
        # Call Grok for final analysis
        try:
            response = self.client.chat.completions.create(
                model=GROK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT + "\n\nThis is the FINAL summary. Be comprehensive and include all key points from the entire meeting."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.3,
                max_tokens=2000,  # Allow longer response for final summary
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            analysis = json.loads(response_text)
            
            # Ensure all keys exist
            required_keys = ["summary", "action_items", "it_insights", "key_decisions"]
            for key in required_keys:
                if key not in analysis:
                    analysis[key] = [] if key != "summary" else "No summary available"
            
            print("✓ Final summary generated")
            return analysis
            
        except Exception as e:
            print(f"❌ Error generating final summary: {e}")
            return self._get_error_response(str(e))
    
    def _get_error_response(self, error_msg: str) -> Dict:
        """Return error response in expected format."""
        return {
            "summary": f"Error: {error_msg}",
            "action_items": [],
            "it_insights": [],
            "key_decisions": []
        }
    
    def get_stats(self) -> Dict:
        """Get meeting statistics."""
        return {
            "duration_minutes": (datetime.now() - self.start_time).seconds // 60,
            "word_count": self.word_count,
            "summary_count": len(self.rolling_summaries),
            "transcript_segments": len(self.full_transcript)
        }


# ============================================================================
# TESTING CODE - Run this file directly to test Grok API
# ============================================================================
from dotenv import load_dotenv
if __name__ == "__main__":
    """
    Test the summarizer independently.
    Run: python summarizer.py
    
    Make sure to set XAI_API_KEY environment variable first:
    export XAI_API_KEY="your_key_here"  # Linux/Mac
    set XAI_API_KEY=your_key_here       # Windows
    """
    
    import os
    load_dotenv()

    
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ Error: XAI_API_KEY environment variable not set")
        print("Get your API key from: https://console.x.ai")
        exit(1)
    
    print("\n" + "="*60)
    print("SUMMARIZER TEST")
    print("="*60)
    
    # Create summarizer
    summarizer = MeetingSummarizer(api_key)
    
    # Simulate meeting transcript
    test_transcript = [
        "Hello everyone, let's start our sprint planning meeting.",
        "We need to discuss the API integration for the new feature.",
        "Bob, can you take the lead on implementing the REST endpoints by Friday?",
        "Sure, I can do that. Should we use GraphQL or REST?",
        "Let's go with REST for now. It's simpler for this use case.",
        "Sarah, please review the database schema and suggest optimizations.",
        "We should also consider caching to improve performance.",
        "Good point. Let's use Redis for caching the frequently accessed data.",
        "I'll create a technical document outlining the architecture by Wednesday.",
        "Great! Let's meet again on Thursday to review progress."
    ]
    
    print("\nSimulating meeting transcript...\n")
    
    for i, segment in enumerate(test_transcript):
        print(f"[{i+1}] {segment}")
        analysis = summarizer.add_transcript(segment)
        
        if analysis:
            print("\n" + "-"*60)
            print("ANALYSIS:")
            print(json.dumps(analysis, indent=2))
            print("-"*60 + "\n")
    
    # Get final summary
    print("\nGenerating final summary...\n")
    final = summarizer.get_final_summary()
    
    print("="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(json.dumps(final, indent=2))
    
    # Show stats
    stats = summarizer.get_stats()
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")