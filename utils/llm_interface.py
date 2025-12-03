"""
LLM Interface Class
A minimal interface for interacting with Large Language Models.
"""

import os
from typing import Optional
from abc import ABC, abstractmethod
import yaml
from pathlib import Path
from openai import RateLimitError
import openai
import re



# Config.yaml paths
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
print(f"Loading config from: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def clean_llm_output(code: str) -> str:
  """
  Remove Markdown code fences and language tags like ```c or ```cpp from LLM output.
  """
  code = re.sub(r"^```[a-zA-Z0-9]*\s*", "", code.strip())  # remove opening ```c or ```cpp
  code = re.sub(r"```$", "", code.strip())  # remove closing ```
  return code.strip()


class LLMInterface(ABC):
    """Abstract base class for LLM interactions."""
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens in the response
            api_key: API key for the service (if None, reads from environment)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class OpenAIInterface(LLMInterface):
    """
    Interface for OpenAI models (GPT-3.5, GPT-4, etc.)
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

          
          
    def generate(self, prompt: str) -> str:
        """
        Generate response using OpenAI API.
        """
        messages = [{"role": "user", "content": prompt}]
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        try:
          response = self.client.chat.completions.create(**params)
        except Exception as e:
          raise e

          
        return clean_llm_output(response.choices[0].message.content)


class GeminiInterface(LLMInterface):
    """
    Interface for Google Gemini models
    """
    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens, api_key)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            generation_config = {
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                generation_config["max_output_tokens"] = self.max_tokens
          
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response using Google Gemini API.
        Handles empty or filtered responses safely.
        """
        try:
            
            response = self.model.generate_content(prompt)
            
            # Some responses might not have .text even if generation succeeded.
            if not hasattr(response, "candidates") or not response.candidates:
                print("No candidates returned from Gemini. Retrying once...")
                response = self.model.generate_content(prompt)
            
            # Still no valid candidate → return empty safely.
            if not hasattr(response, "candidates") or not response.candidates:
                print("No valid candidates after retry.")
                return ""
            
            # Extract text safely from first candidate.
            candidate = response.candidates[0]
            if not candidate or not candidate.content.parts:
                print("Candidate has no text parts.")
                return ""
            
            # Join all text parts safely.
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            
            if not text_parts:
                print("No text content found in response parts.")
                return ""

            return clean_llm_output("\n".join(text_parts))
        
        except Exception as e:
            print(f"[GeminiInterface] Error: {e}")
            return ""



def create_llm_interface(
    provider: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMInterface:
    """
    Factory function to create an LLM interface.
    """
    providers = {
        'openai': OpenAIInterface,
        'gemini': GeminiInterface,
    }
    
    if provider.lower() not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: {list(providers.keys())}"
        )
    
    interface_class = providers[provider.lower()]
  
    
    return interface_class(model_name=model_name, api_key=api_key)

