from transformers import pipeline, GPT2Tokenizer
import os
from dotenv import load_dotenv

load_dotenv()

class ContentPlanningAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2")  # Use Hugging Face's text-generation pipeline
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load the tokenizer

    def create_outline(self, topic, information):
        prompt = f"Create a detailed blog outline for the topic: {topic}. Use the following information: {information}"
        
        # Tokenize the prompt and truncate if necessary
        tokens = self.tokenizer.encode(prompt, truncation=True, max_length=512, return_tensors="pt")
        truncated_prompt = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        
        try:
            generated_text = self.generator(
                truncated_prompt,
                max_new_tokens=200,  # Limit the number of tokens generated
                num_return_sequences=1,
                truncation=True  # Ensure truncation is applied
            )[0]["generated_text"]
            return generated_text
        except Exception as e:
            print(f"Error generating outline: {e}")
            return ""  # Return an empty string if generation fails