import openai
import json
import numpy as np
import os
import logging
import math
import re
import exifread
import cv2
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeepfakeTextDetector:
    def _init_(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        logging.info("DeepfakeTextDetector initialized successfully.")
    
    def analyze_text(self, text: str) -> dict:
        system_prompt = """
        You are an AI text detection system that determines whether a given text is AI-generated or human-written.
        Your analysis should focus on:
        - Sentence structure: AI text often has predictable structures with low variation.
        - Word choice: AI models tend to use formal, neutral, and statistically common words.
        - Perplexity & entropy: Lower perplexity suggests AI-generated text (more predictable sequences).
        - Lexical diversity: AI text tends to reuse common words, while humans use a wider range.
        - Logical coherence: AI-generated text may be logically consistent but lacks deep reasoning.
        - Repetitive phrases: AI text may repeat ideas without truly expanding on them.
        Provide a structured JSON output:
        {
            "confidence_score": float (between 0 and 1),
            "key_indicators": list of specific reasons why the text may/may not be AI-generated,
            "explanation": string (clear and logically sound explanation),
            "language_style": string (e.g., "formal", "concise", "verbose"),
            "sentence_complexity": string (e.g., "simple", "moderate", "complex")
        }
        Only return valid JSON.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text to analyze: {text}"}
                ],
                temperature=0.2
            )
            
            analysis = self._parse_analysis(response.choices[0].message.content)
            stats = self._improved_statistical_analysis(text)
            analysis.update(stats)

            entropy = stats["statistical_metrics"]["entropy"]
            perplexity = stats["statistical_metrics"]["perplexity"]
            bigram_repetition = stats["statistical_metrics"]["bigram_repetition"]
            trigram_repetition = stats["statistical_metrics"]["trigram_repetition"]

            confidence = analysis["confidence_score"]

            if entropy < 3.5:
                confidence += 0.1
            if perplexity < 20:
                confidence += 0.1
            if bigram_repetition < 0.6:
                confidence += 0.05
            if trigram_repetition < 0.5:
                confidence += 0.05

            confidence = min(max(confidence, 0), 1)
            if confidence > 0.5:
                confidence = min(1, confidence * 1.3)  
            else:
                confidence = max(0, confidence * 0.7)  
            analysis["confidence_score"] = round(1 - confidence, 3)

            logging.info(f"Final Confidence Score: {analysis['confidence_score']}")
            return analysis
        
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _improved_statistical_analysis(self, text: str) -> dict:
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        word_lengths = [len(word) for word in words]
        
        word_freq = Counter(words)
        total_words = len(words)
        entropy = -sum((freq / total_words) * math.log2(freq / total_words) for freq in word_freq.values()) if words else 0
        perplexity = math.exp(entropy) if words else 0
        
        lexical_diversity = len(set(words)) / total_words if words else 0
        
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        bigram_repetition = len(set(bigrams)) / len(bigrams) if bigrams else 0
        trigram_repetition = len(set(trigrams)) / len(trigrams) if trigrams else 0
        
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)

        return {
            "statistical_metrics": {
                "word_count": total_words,
                "avg_word_length": avg_word_length,
                "sentence_count": len(sentences),
                "avg_sentence_length": avg_sentence_length,
                "unique_words_ratio": lexical_diversity,
                "entropy": entropy,
                "perplexity": perplexity,
                "readability_score": readability_score,
                "bigram_repetition": bigram_repetition,
                "trigram_repetition": trigram_repetition
            }
        }
    
    def _parse_analysis(self, response_text: str) -> dict:
        try:
            analysis = json.loads(response_text.strip())
            analysis["timestamp"] = datetime.now().isoformat()
            analysis["version"] = "1.4"
            return analysis
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response: {e}")
            return {"error": "Invalid response format", "raw_analysis": response_text}

if _name_ == "_main_":
    text_detector = DeepfakeTextDetector(api_key=api_key)
    
    while True:
        user_choice = input("\nEnter 'text' for deepfake analysis (or 'exit' to quit): ").lower()
        if user_choice == "exit":
            break
        elif user_choice == "text":
            text = input("Enter text to analyze: ")
            result = text_detector.analyze_text(text)
            print("\nConfidence Score:", result["confidence_score"])
        else:
            print("Invalid option.")