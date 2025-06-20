#!/usr/bin/env python3
"""
AI Brand Recommendation Script
Automatically processes product CSV and gets brand recommendations from multiple AI models
"""

import csv
import json
import time
import schedule
import threading
import os
from datetime import datetime
from pathlib import Path
import requests
import openai
from anthropic import Anthropic
import pandas as pd
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brand_recommender.log'),
        logging.StreamHandler()
    ]
)

class BrandRecommender:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # API Keys from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # Validate API keys
        self._validate_api_keys()
        
        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
        
        # Configuration
        self.input_file = "products_input.csv"  # Change this to your input file path
        self.output_dir = "brand_recommendations"
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # AI Models to use
        self.models = {
            "gpt-4-turbo": {"provider": "openai", "model": "gpt-4-turbo"},
            "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
            "claude-3.5-sonnet": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            "claude-sonnet-4": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            "deepseek-chat": {"provider": "deepseek", "model": "deepseek-chat"},
            "deepseek-coder": {"provider": "deepseek", "model": "deepseek-coder"}
        }

    def _validate_api_keys(self):
        """Validate that all required API keys are present"""
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            missing_keys.append("ANTHROPIC_API_KEY")
        if not self.deepseek_api_key:
            missing_keys.append("DEEPSEEK_API_KEY")
            
        if missing_keys:
            error_msg = f"Missing required API keys: {', '.join(missing_keys)}\n"
            error_msg += "Please set them in your .env file or environment variables."
            logging.error(error_msg)
            raise ValueError(error_msg)

    def create_prompt(self, product):
        """Create the prompt for brand recommendation"""
        return f"""Please recommend the top 5 brands for the product: "{product}"

Requirements:
- Focus on well-known, reputable brands
- Consider quality, popularity, and market presence
- Provide brands that are currently available in the market
- Order them from most recommended (1) to least recommended (5)

Please respond in this exact JSON format:
{{
    "product": "{product}",
    "brands": [
        {{"rank": 1, "brand": "Brand Name 1", "reason": "Brief reason"}},
        {{"rank": 2, "brand": "Brand Name 2", "reason": "Brief reason"}},
        {{"rank": 3, "brand": "Brand Name 3", "reason": "Brief reason"}},
        {{"rank": 4, "brand": "Brand Name 4", "reason": "Brief reason"}},
        {{"rank": 5, "brand": "Brand Name 5", "reason": "Brief reason"}}
    ]
}}"""

    def query_openai(self, model, prompt):
        """Query OpenAI models"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI {model} error: {e}")
            return None

    def query_anthropic(self, model, prompt):
        """Query Anthropic models"""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic {model} error: {e}")
            return None

    def query_deepseek(self, model, prompt):
        """Query DeepSeek models"""
        try:
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"DeepSeek {model} error: {e}")
            return None

    def query_model(self, model_name, model_info, prompt):
        """Query a specific model based on provider"""
        provider = model_info["provider"]
        model = model_info["model"]
        
        if provider == "openai":
            return self.query_openai(model, prompt)
        elif provider == "anthropic":
            return self.query_anthropic(model, prompt)
        elif provider == "deepseek":
            return self.query_deepseek(model, prompt)
        else:
            logging.error(f"Unknown provider: {provider}")
            return None

    def parse_response(self, response_text):
        """Parse JSON response from AI model"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logging.warning("No JSON found in response")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return None

    def process_product(self, product):
        """Process a single product with all AI models"""
        logging.info(f"Processing product: {product}")
        results = []
        
        prompt = self.create_prompt(product)
        
        for model_name, model_info in self.models.items():
            logging.info(f"Querying {model_name}...")
            
            response = self.query_model(model_name, model_info, prompt)
            if response:
                parsed_response = self.parse_response(response)
                if parsed_response and "brands" in parsed_response:
                    for brand_info in parsed_response["brands"]:
                        results.append({
                            "product": product,
                            "model": model_name,
                            "rank": brand_info.get("rank", ""),
                            "brand": brand_info.get("brand", ""),
                            "reason": brand_info.get("reason", ""),
                            "timestamp": datetime.now().isoformat()
                        })
                else:
                    logging.warning(f"Failed to parse response from {model_name}")
            
            # Small delay between API calls
            time.sleep(1)
        
        return results

    def read_products_csv(self):
        """Read products from input CSV file"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                products = []
                for row in reader:
                    if row and row[0].strip():  # Skip empty rows
                        products.append(row[0].strip())
                
                # Remove header if it exists
                if products and products[0].lower() in ['product', 'products', 'product_name']:
                    products = products[1:]
                
                return products
        except FileNotFoundError:
            logging.error(f"Input file {self.input_file} not found!")
            return []
        except Exception as e:
            logging.error(f"Error reading CSV: {e}")
            return []

    def save_results_csv(self, results):
        """Save results to CSV file"""
        if not results:
            logging.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/brand_recommendations_{timestamp}.csv"
        
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logging.info(f"Results saved to {output_file}")
            
            # Also save a summary with aggregated recommendations
            self.save_summary_csv(results, timestamp)
            
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")

    def save_summary_csv(self, results, timestamp):
        """Save aggregated summary of recommendations"""
        summary_file = f"{self.output_dir}/brand_summary_{timestamp}.csv"
        
        try:
            df = pd.DataFrame(results)
            
            # Aggregate recommendations by product and brand
            summary = df.groupby(['product', 'brand']).agg({
                'model': 'count',  # How many models recommended this brand
                'rank': 'mean',    # Average rank
                'reason': 'first'  # Take first reason
            }).rename(columns={'model': 'model_count', 'rank': 'avg_rank'}).reset_index()
            
            # Sort by product and model count (most recommended first)
            summary = summary.sort_values(['product', 'model_count', 'avg_rank'], ascending=[True, False, True])
            
            summary.to_csv(summary_file, index=False, encoding='utf-8')
            logging.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logging.error(f"Error saving summary: {e}")

    def run_analysis(self):
        """Main function to run the brand recommendation analysis"""
        logging.info("Starting brand recommendation analysis...")
        
        # Read products from CSV
        products = self.read_products_csv()
        if not products:
            logging.error("No products found in input CSV")
            return
        
        logging.info(f"Found {len(products)} products to analyze")
        
        # Process each product
        all_results = []
        for i, product in enumerate(products, 1):
            logging.info(f"Processing {i}/{len(products)}: {product}")
            results = self.process_product(product)
            all_results.extend(results)
            
            # Small delay between products
            time.sleep(2)
        
        # Save results
        if all_results:
            self.save_results_csv(all_results)
            logging.info(f"Analysis completed! Processed {len(products)} products with {len(all_results)} total recommendations.")
        else:
            logging.error("No results generated")

    def create_sample_input(self):
        """Create a sample input CSV file"""
        sample_products = [
            "Smartphone",
            "Laptop",
            "Running Shoes",
            "Coffee Maker",
            "Headphones"
        ]
        
        with open(self.input_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["product"])  # Header
            for product in sample_products:
                writer.writerow([product])
        
        logging.info(f"Sample input file created: {self.input_file}")

def run_scheduler():
    """Run the scheduled tasks"""
    recommender = BrandRecommender()
    
    # Check if input file exists, create sample if not
    if not Path(recommender.input_file).exists():
        print(f"Input file {recommender.input_file} not found. Creating sample file...")
        recommender.create_sample_input()
        print("Please edit the input file with your products and restart the script.")
        return
    
    recommender.run_analysis()

def main():
    """Main function with scheduling options"""
    print("AI Brand Recommendation Script")
    print("=" * 40)
    
    # Setup
    recommender = BrandRecommender()
    
    # Check if input file exists
    if not Path(recommender.input_file).exists():
        print(f"Input file {recommender.input_file} not found. Creating sample file...")
        recommender.create_sample_input()
        print("Please edit the input file with your products and restart the script.")
        return
    
    print("\nScheduling Options:")
    print("1. Run once now")
    print("2. Run every hour")
    print("3. Run daily at 9 AM")
    print("4. Run weekly on Monday at 9 AM")
    print("5. Custom schedule (enter cron-like frequency)")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        run_scheduler()
    elif choice == "2":
        schedule.every().hour.do(run_scheduler)
        print("Scheduled to run every hour. Press Ctrl+C to stop.")
    elif choice == "3":
        schedule.every().day.at("09:00").do(run_scheduler)
        print("Scheduled to run daily at 9 AM. Press Ctrl+C to stop.")
    elif choice == "4":
        schedule.every().monday.at("09:00").do(run_scheduler)
        print("Scheduled to run weekly on Monday at 9 AM. Press Ctrl+C to stop.")
    elif choice == "5":
        minutes = input("Enter frequency in minutes (e.g., 60 for hourly): ").strip()
        try:
            minutes = int(minutes)
            schedule.every(minutes).minutes.do(run_scheduler)
            print(f"Scheduled to run every {minutes} minutes. Press Ctrl+C to stop.")
        except ValueError:
            print("Invalid input. Running once now.")
            run_scheduler()
            return
    else:
        print("Invalid choice. Running once now.")
        run_scheduler()
        return
    
    # Keep the scheduler running
    if choice in ["2", "3", "4", "5"]:
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nScheduler stopped.")

if __name__ == "__main__":
    main()
