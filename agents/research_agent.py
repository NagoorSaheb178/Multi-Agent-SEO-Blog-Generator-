from transformers import pipeline, GPT2Tokenizer
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time

load_dotenv()

class ResearchAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2")  # Use Hugging Face's text-generation pipeline
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load the tokenizer

    def find_trending_topics(self):
        try:
            # Use a valid URL for scraping trending HR topics
            url = "https://www.shrm.org"  # Example: Society for Human Resource Management (SHRM)
            response = requests.get(url, timeout=10)  # Add a timeout to avoid hanging
            response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404, 500)
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract trending topics (adjust the HTML tags as needed)
            topics = [topic.text for topic in soup.find_all("h2", class_="title")]  # Example: Find all <h2> tags with class "title"
            
            # If no topics are found, use the LLM to generate topics
            if not topics:
                print("No topics found. Generating topics using Hugging Face...")
                prompt = "Generate a list of 5 trending HR topics:"
                
                # Tokenize the prompt and truncate if necessary
                tokens = self.tokenizer.encode(prompt, truncation=True, max_length=512, return_tensors="pt")
                truncated_prompt = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
                
                generated_text = self.generator(
                    truncated_prompt,
                    max_new_tokens=100,  # Limit the number of tokens generated
                    num_return_sequences=1,
                    truncation=True  # Ensure truncation is applied
                )[0]["generated_text"]
                topics = generated_text.split("\n")[:5]  # Split the LLM output into a list of topics
            
            return topics[:5]  # Return top 5 trending topics
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trending topics: {e}")
            return []  # Return an empty list if there's an error

    def collect_information(self, topic):
        prompt = f"Find detailed information about the HR topic: {topic}"
        
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
            print(f"Error generating information: {e}")
            return ""  # Return an empty string if generation fails