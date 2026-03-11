import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json

class FallbackProcessingError(Exception):
    def __init__(self, message: str, fallback_result: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.fallback_result = fallback_result

class QuotaExceededError(Exception):
    pass

def get_api_key_from_env() -> str:
    return os.getenv("GEMINI_API_KEY", "")

class AIService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.max_chars = 20000  # Limit input text to avoid token limit errors
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text if it exceeds the character limit"""
        if len(text) > self.max_chars:
            print(f"Warning: Text truncated from {len(text)} to {self.max_chars} characters")
            return text[:self.max_chars]
        return text
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process the OCR text to extract clean notes and tasks"""
        try:
            # Truncate text if too long
            text = self._truncate_text(text)
            print(f"Processing text ({len(text)} chars): {text[:200]}...")
            
            prompt = f"""Analyze this text and extract:
1. Clean, readable notes (fix OCR errors, format properly)
2. Any tasks/action items (buy, call, email, schedule, finish, complete, etc.)

Text: {text}

Return JSON:
{{"clean_notes": "text here", "tasks": ["task1", "task2"], "model": "gemini-2.5-flash"}}
If no tasks found, use empty list."""
            
            print(f"Sending request to Gemini API...")
            response = self.model.generate_content(prompt)
            result_text = response.text
            print(f"AI response ({len(result_text or '')} chars): {(result_text or '')[:500]}...")
            
            if result_text is None:
                raise ValueError("No response text from AI model")
            
            # Try to parse as JSON
            try:
                result = json.loads(result_text)
                result['raw_response'] = result_text
                print(f"✓ JSON parsed successfully")
                return result
            except json.JSONDecodeError as je:
                print(f"JSON parse error: {je}, attempting fallback")
                # Fallback parsing
                clean_notes = result_text.split('"clean_notes": "')[1].split('"')[0] if '"clean_notes": "' in result_text else text
                tasks_str = result_text.split('"tasks": [')[1].split(']')[0] if '"tasks": [' in result_text else ""
                tasks = [t.strip().strip('"').strip("'") for t in tasks_str.split(',') if t.strip()]
                return {
                    "clean_notes": clean_notes,
                    "tasks": tasks,
                    "model": "gemini-2.5-flash",
                    "raw_response": result_text
                }
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Processing failed: {error_msg}")
            
            if "quota" in error_msg.lower():
                raise QuotaExceededError("API quota exceeded. Please try again later.")
            elif "content" in error_msg.lower() or "safety" in error_msg.lower():
                raise FallbackProcessingError(
                    "Content blocked by safety filter. Try with different content.",
                    {"clean_notes": text, "tasks": [], "model": "gemini-2.5-flash", "raw_response": error_msg}
                )
            
            fallback_result = {
                "clean_notes": text,
                "tasks": [],
                "model": "gemini-2.5-flash",
                "raw_response": f"Error: {error_msg}"
            }
            raise FallbackProcessingError(f"AI processing failed: {error_msg}", fallback_result)