import streamlit as st
import json
import os
import random
import re
from typing import Dict, Any

# Configure the page
st.set_page_config(
    page_title="MCQ Generator", 
    page_icon="📝", 
    layout="wide"
)

class SimpleMCQGenerator:
    """Simple rule-based MCQ generator that works without any APIs"""
    
    def __init__(self):
        self.question_templates = [
            "What is the main concept related to {term}?",
            "According to the text, {term} is described as:",
            "Which of the following best describes {term}?",
            "In the context of {subject}, {term} refers to:",
            "What can be concluded about {term} from the text?"
        ]
        
        self.distractor_templates = [
            "It is not mentioned in the text",
            "It is completely unrelated to {subject}",
            "It was discovered in ancient times",
            "It is a fictional concept",
            "It has no practical applications"
        ]
    
    def extract_key_terms(self, text: str, min_length: int = 4) -> list:
        """Extract important terms from text"""
        # Clean and split text
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        
        # Filter out common words
        common_words = {
            'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been', 
            'have', 'their', 'said', 'each', 'which', 'them', 'than', 'many',
            'some', 'what', 'would', 'make', 'like', 'into', 'time', 'very',
            'when', 'much', 'know', 'take', 'good', 'just', 'first', 'well',
            'also', 'after', 'work', 'life', 'only', 'new', 'way', 'may',
            'say', 'come', 'could', 'now', 'over', 'such', 'our', 'out',
            'other', 'more', 'get', 'these', 'give', 'most', 'us'
        }
        
        key_terms = [word for word in words if word.lower() not in common_words]
        return list(set(key_terms))  # Remove duplicates
    
    def create_question(self, text: str, term: str, subject: str) -> Dict[str, Any]:
        """Create a single MCQ"""
        
        # Find sentence containing the term
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        relevant_sentence = ""
        
        for sentence in sentences:
            if term.lower() in sentence.lower():
                relevant_sentence = sentence
                break
        
        if not relevant_sentence:
            relevant_sentence = sentences[0] if sentences else text[:100]
        
        # Create question
        question_template = random.choice(self.question_templates)
        question = question_template.format(term=term, subject=subject)
        
        # Create correct answer (extract relevant info)
        correct_answer = relevant_sentence
        if len(correct_answer) > 80:
            correct_answer = correct_answer[:77] + "..."
        
        # Create distractors
        distractors = []
        for template in random.sample(self.distractor_templates, 3):
            distractor = template.format(subject=subject, term=term)
            distractors.append(distractor)
        
        # Combine options
        all_options = [correct_answer] + distractors
        random.shuffle(all_options)
        
        # Find correct position
        correct_pos = ["a", "b", "c", "d"][all_options.index(correct_answer)]
        
        return {
            "mcq": question,
            "options": {
                "a": all_options[0],
                "b": all_options[1], 
                "c": all_options[2],
                "d": all_options[3]
            },
            "correct": correct_pos
        }
    
    def generate_mcqs(self, text: str, number: int, subject: str) -> str:
        """Generate multiple MCQs"""
        
        key_terms = self.extract_key_terms(text)
        
        if len(key_terms) < number:
            # If not enough terms, use sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            while len(key_terms) < number and sentences:
                sentence = sentences.pop(0)
                terms_in_sentence = self.extract_key_terms(sentence)
                key_terms.extend(terms_in_sentence[:2])  # Add up to 2 terms per sentence
        
        # Take only the number we need
        selected_terms = key_terms[:number]
        
        mcqs = {}
        for i, term in enumerate(selected_terms, 1):
            mcq = self.create_question(text, term, subject)
            mcqs[str(i)] = mcq
        
        return json.dumps(mcqs, indent=2)

def load_text_file(uploaded_file):
    """Load text from uploaded file"""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                import io
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                st.error("PyPDF2 not installed. Please install it to read PDF files: pip install PyPDF2")
                return None
        else:
            st.error("Unsupported file type")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.title("📝 MCQ Generator")
    st.markdown("Generate Multiple Choice Questions from your text content")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a PDF or TXT file
        2. Specify the number of MCQs (3-50)
        3. Enter the subject area
        4. Set the complexity level
        5. Click 'Create MCQs' to generate
        """)
        
        st.header("📁 Supported Formats")
        st.markdown("""
        • PDF files (.pdf)
        • Text files (.txt)
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=['pdf', 'txt'],
            help="Limit 200MB per file • PDF, TXT"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with col2:
        subject = st.text_input("Subject", value="Biology", help="Enter the subject area")
        complexity = st.selectbox("🎯 Complexity Level", ["simple", "medium", "complex"])
    
    # Number of MCQs
    number = st.number_input(
        "📝 Number of MCQs", 
        min_value=3, 
        max_value=50, 
        value=5,
        help="Choose between 3-50 questions"
    )
    
    # Generate button
    if st.button("🚀 Create MCQs", type="primary", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processing results..."):
                # Load text
                text = load_text_file(uploaded_file)
                
                if text:
                    # Generate MCQs
                    try:
                        generator = SimpleMCQGenerator()
                        quiz_json = generator.generate_mcqs(text, number, subject)
                        
                        # Parse and display results
                        quiz_data = json.loads(quiz_json)
                        
                        st.success("✅ MCQs generated successfully!")
                        
                        # Display quiz
                        st.header("📋 Generated Quiz")
                        
                        for q_num, q_data in quiz_data.items():
                            with st.expander(f"Question {q_num}: {q_data['mcq']}", expanded=True):
                                for opt_key, opt_value in q_data['options'].items():
                                    if opt_key == q_data['correct']:
                                        st.markdown(f"**{opt_key.upper()}) {opt_value}** ✅")
                                    else:
                                        st.markdown(f"{opt_key.upper()}) {opt_value}")
                        
                        # Download section
                        st.header("💾 Download Quiz")
                        
                        # Format for download
                        download_text = f"Quiz: {subject}\n"
                        download_text += f"Number of Questions: {len(quiz_data)}\n"
                        download_text += f"Complexity: {complexity}\n"
                        download_text += "="*50 + "\n\n"
                        
                        for q_num, q_data in quiz_data.items():
                            download_text += f"Question {q_num}: {q_data['mcq']}\n"
                            for opt_key, opt_value in q_data['options'].items():
                                marker = " (Correct)" if opt_key == q_data['correct'] else ""
                                download_text += f"{opt_key.upper()}) {opt_value}{marker}\n"
                            download_text += "\n"
                        
                        st.download_button(
                            label="📥 Download Quiz as TXT",
                            data=download_text,
                            file_name=f"{subject}_quiz_{number}questions.txt",
                            mime="text/plain"
                        )
                        
                        # JSON download
                        st.download_button(
                            label="📥 Download as JSON",
                            data=quiz_json,
                            file_name=f"{subject}_quiz_{number}questions.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating MCQs: {str(e)}")
                        st.markdown("💡 Try with a different file or reduce the number of MCQs")
                else:
                    st.error("❌ Could not read the uploaded file")
        else:
            st.warning("⚠️ Please upload a file first")
    
    # Sample text for testing
    with st.expander("🔬 Test with Sample Text", expanded=False):
        sample_text = st.text_area(
            "Enter sample text to test:",
            value="""Artificial Intelligence is a branch of computer science that aims to create intelligent machines. These machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has been particularly successful in image recognition and natural language processing.""",
            height=150
        )
        
        if st.button("Generate from Sample Text"):
            with st.spinner("Generating from sample..."):
                try:
                    generator = SimpleMCQGenerator()
                    quiz_json = generator.generate_mcqs(sample_text, 3, "Computer Science")
                    quiz_data = json.loads(quiz_json)
                    
                    st.success("✅ Sample MCQs generated!")
                    
                    for q_num, q_data in quiz_data.items():
                        with st.expander(f"Question {q_num}: {q_data['mcq']}", expanded=True):
                            for opt_key, opt_value in q_data['options'].items():
                                if opt_key == q_data['correct']:
                                    st.markdown(f"**{opt_key.upper()}) {opt_value}** ✅")
                                else:
                                    st.markdown(f"{opt_key.upper()}) {opt_value}")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()