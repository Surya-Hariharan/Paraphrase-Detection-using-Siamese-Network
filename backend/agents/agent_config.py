"""
Agent Configuration Module
===========================

Manages LLM provider configuration for all agents.
Supports Gemini and Groq providers.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import google.generativeai as genai

# Load environment variables
load_dotenv()


class AgentConfig:
    """Configuration manager for AI agents."""
    
    def __init__(self, provider: str = "gemini", enable_logging: bool = True):
        """
        Initialize agent configuration.
        
        Args:
            provider: LLM provider ("gemini" or "groq")
            enable_logging: Whether to log agent interventions
        """
        self.provider = provider.lower()
        self.enable_logging = enable_logging
        self.log_dir = Path("logs/agents")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env file")
            
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash')
        
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            
            try:
                from groq import Groq
                return Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq package not installed. Run: pip install groq")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            temperature: Generation temperature (0.0 to 1.0)
        
        Returns:
            Generated response text
        """
        try:
            if self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                    )
                )
                return response.text
            
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content
        
        except Exception as e:
            if self.enable_logging:
                self.log_intervention("ERROR", f"LLM generation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def log_intervention(self, agent_name: str, message: str):
        """Log an agent intervention."""
        if not self.enable_logging:
            return
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = self.log_dir / f"{agent_name.lower()}.log"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
