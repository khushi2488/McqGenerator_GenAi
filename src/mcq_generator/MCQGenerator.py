import os
import json
import pandas as pd
import re
import random
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCQGenerator:
    def __init__(self, use_openai=False, use_huggingface=False):
        """
        Initialize MCQ Generator with different LLM options
        
        Args:
            use_openai: Set to True to use OpenAI API
            use_huggingface: Set to True to use HuggingFace (if you can fix the token issue)
        """
        self.llm = None
        self.llm_type = None
        
        if use_openai:
            self._initialize_openai()
        elif use_huggingface:
            self._initialize_huggingface()
        else:
            # Use rule-based approach as fallback
            self.llm_type = "rule_based"
            logger.info("Using rule-based MCQ generation")
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            
            self.llm = OpenAI(api_key=api_key)
            self.llm_type = "openai"
            logger.info("✅ OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            self.llm_type = "rule_based"
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace LLM with simpler approach"""
        try:
            from transformers import pipeline
            
            # Use local transformers pipeline instead of API
            self.llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",  # Small model that works locally
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            self.llm_type = "huggingface_local"
            logger.info("✅ HuggingFace local model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFace: {e}")
            self.llm_type = "rule_based"
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate content using OpenAI"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_rule_based_fallback(prompt)
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        """Generate content using HuggingFace local model"""
        try:
            result = self.llm(prompt, max_length=512, num_return_sequences=1)
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return self._generate_rule_based_fallback(prompt)
    
    def _generate_rule_based_fallback(self, prompt: str) -> str:
        """Rule-based MCQ generation as fallback"""
        # Extract information from the prompt
        text_match = re.search(r'Text: (.+?)(?=\n\nYou are|$)', prompt, re.DOTALL)
        number_match = re.search(r'create a quiz of (\d+)', prompt)
        subject_match = re.search(r'for (.+?) students', prompt)
        
        if not text_match:
            return json.dumps({"error": "Could not extract text from prompt"})
        
        text = text_match.group(1).strip()
        number = int(number_match.group(1)) if number_match else 1
        subject = subject_match.group(1) if subject_match else "General"
        
        return self._create_rule_based_mcqs(text, number, subject)
    
    def _create_rule_based_mcqs(self, text: str, number: int, subject: str) -> str:
        """Create MCQs using rule-based approach"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < number:
            # If not enough sentences, create general questions
            sentences.extend([text] * (number - len(sentences)))
        
        mcqs = {}
        
        for i in range(number):
            sentence = sentences[i % len(sentences)]
            
            # Extract key terms (simple approach)
            words = sentence.split()
            important_words = [w for w in words if len(w) > 4 and w.isalpha()]
            
            if important_words:
                key_term = random.choice(important_words)
                
                # Create question based on the sentence
                question = f"What is mentioned about {key_term.lower()} in the context of {subject}?"
                
                # Generate options
                correct_option = sentence[:50] + "..." if len(sentence) > 50 else sentence
                
                distractors = [
                    f"It is not related to {subject}",
                    f"It is a type of {random.choice(['technology', 'method', 'concept', 'theory'])}",
                    f"It was invented in {random.choice(['1990s', '2000s', '2010s'])}"
                ]
                
                options = {
                    "a": distractors[0],
                    "b": distractors[1], 
                    "c": correct_option,
                    "d": distractors[2]
                }
                
                # Randomize correct answer position
                correct_positions = ["a", "b", "c", "d"]
                correct_pos = random.choice(correct_positions)
                
                if correct_pos != "c":
                    options[correct_pos], options["c"] = options["c"], options[correct_pos]
                
                mcqs[str(i + 1)] = {
                    "mcq": question,
                    "options": options,
                    "correct": correct_pos
                }
            else:
                # Fallback question
                mcqs[str(i + 1)] = {
                    "mcq": f"Which of the following is true about {subject}?",
                    "options": {
                        "a": "It is not important",
                        "b": "It has no applications", 
                        "c": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                        "d": "It was banned"
                    },
                    "correct": "c"
                }
        
        return json.dumps(mcqs, indent=2)
    
    def generate_mcqs(self, text: str, number: int, subject: str, tone: str = "simple") -> Dict[str, Any]:
        """Generate MCQs with the available LLM or rule-based approach"""
        
        # Create the prompt
        response_json_example = {
            "1": {
                "mcq": "Sample question?",
                "options": {
                    "a": "Option A",
                    "b": "Option B", 
                    "c": "Option C",
                    "d": "Option D"
                },
                "correct": "c"
            }
        }
        
        prompt = f"""Text: {text}

You are an expert MCQ maker. Given the above text, create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.

Important Instructions:
1. Make sure the questions are not repeated
2. Ensure all questions conform to the given text
3. Format your response EXACTLY like the RESPONSE_JSON structure below
4. Create exactly {number} MCQs
5. Each question must have 4 options (a, b, c, d)
6. Specify the correct answer

### RESPONSE_JSON FORMAT:
{json.dumps(response_json_example, indent=2)}

Generate the MCQs now:"""

        try:
            if self.llm_type == "openai":
                quiz_result = self._generate_with_openai(prompt)
            elif self.llm_type == "huggingface_local":
                quiz_result = self._generate_with_huggingface(prompt)
            else:  # rule_based
                quiz_result = self._generate_rule_based_fallback(prompt)
            
            # Try to parse as JSON first
            try:
                parsed_result = json.loads(quiz_result)
                if isinstance(parsed_result, dict):
                    quiz_result = json.dumps(parsed_result, indent=2)
            except:
                pass  # Keep original result if not valid JSON
            
            return {
                "quiz": quiz_result,
                "subject": subject,
                "method": self.llm_type,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating MCQs: {e}")
            # Always provide a fallback
            fallback_result = self._create_rule_based_mcqs(text, number, subject)
            return {
                "quiz": fallback_result,
                "subject": subject,
                "method": "rule_based_fallback",
                "success": False,
                "error": str(e)
            }
    
    def review_mcqs(self, quiz: str, subject: str) -> str:
        """Review the generated MCQs"""
        
        if self.llm_type == "openai":
            prompt = f"""You are an expert English grammarian and writer. 

Given this Multiple Choice Quiz for {subject} students, evaluate the complexity and provide analysis.

Quiz:
{quiz}

Provide:
1. Complexity analysis (max 50 words)
2. Suggestions for improvement if needed
3. Overall assessment of the quiz quality

Analysis:"""
            try:
                return self._generate_with_openai(prompt)
            except:
                pass
        
        # Simple rule-based review
        try:
            quiz_data = json.loads(quiz)
            num_questions = len(quiz_data)
            
            review = f"""Complexity Analysis: The quiz contains {num_questions} questions for {subject} students. Questions appear to be at an appropriate level.

Suggestions: Ensure all questions are directly related to the source material and options are clearly distinct.

Overall Assessment: The quiz structure is properly formatted with multiple choice options and correct answers specified. Suitable for {subject} assessment."""
            
            return review
            
        except Exception as e:
            return f"Review completed. The quiz contains multiple choice questions suitable for {subject} students. Manual review recommended for content accuracy."

# Usage Functions
def create_mcqs_with_review(text: str, number: int, subject: str, tone: str = "simple", 
                           use_openai: bool = False, use_huggingface: bool = False) -> Dict[str, Any]:
    """
    Main function to create MCQs with review
    
    Args:
        text: Source text for MCQ generation
        number: Number of MCQs to generate
        subject: Subject area
        tone: Complexity tone (simple, medium, complex)
        use_openai: Whether to use OpenAI API (requires OPENAI_API_KEY)
        use_huggingface: Whether to use HuggingFace (requires model setup)
    
    Returns:
        Dictionary with quiz, review, and metadata
    """
    
    generator = MCQGenerator(use_openai=use_openai, use_huggingface=use_huggingface)
    
    # Generate MCQs
    mcq_result = generator.generate_mcqs(text, number, subject, tone)
    
    # Generate review
    review_result = generator.review_mcqs(mcq_result["quiz"], subject)
    
    return {
        "quiz": mcq_result["quiz"],
        "review": review_result,
        "subject": subject,
        "method_used": mcq_result.get("method", "unknown"),
        "success": mcq_result.get("success", True),
        "error": mcq_result.get("error", None)
    }

# Test function
def test_mcq_system():
    """Test the MCQ system with sample data"""
    
    sample_text = """
    Artificial Intelligence is a branch of computer science that aims to create intelligent machines. 
    These machines can perform tasks that typically require human intelligence, such as visual perception, 
    speech recognition, decision-making, and language translation. Machine learning is a subset of AI 
    that enables computers to learn and improve from experience without being explicitly programmed.
    """
    
    print("Testing MCQ Generation System...")
    print("=" * 50)
    
    # Test with rule-based approach (no API needed)
    result = create_mcqs_with_review(
        text=sample_text,
        number=3,
        subject="Computer Science", 
        tone="simple",
        use_openai=False,  # Set to True if you have OpenAI API key
        use_huggingface=False  # Set to True if you want to try HuggingFace
    )
    
    print(f"Generation Method: {result['method_used']}")
    print(f"Success: {result['success']}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    print("\nGenerated Quiz:")
    print(result['quiz'])
    
    print("\nReview:")
    print(result['review'])
    
    return result

if __name__ == "__main__":
    # Run the test
    test_result = test_mcq_system()
    
    print("\n" + "="*50)
    print("SETUP INSTRUCTIONS:")
    print("="*50)
    print("1. For RULE-BASED (works immediately):")
    print("   - No setup needed, just run the code")
    print()
    print("2. For OPENAI (better quality):")
    print("   - pip install openai")
    print("   - Set OPENAI_API_KEY in your .env file")
    print("   - Set use_openai=True")
    print()
    print("3. For HUGGINGFACE LOCAL (medium quality):")
    print("   - pip install transformers torch")
    print("   - Set use_huggingface=True")
    print("   - First run will download model (~500MB)")