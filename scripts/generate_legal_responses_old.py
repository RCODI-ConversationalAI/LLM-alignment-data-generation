import openai
import pandas as pd
from pathlib import Path
import json
import time
from typing import List, Dict
from sklearn.model_selection import train_test_split
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import os

# Check OpenAI version
OPENAI_VERSION = pkg_resources.get_distribution("openai").version
if OPENAI_VERSION != "0.28.0":
    raise ImportError(f"Please install openai==0.28.0. Current version: {OPENAI_VERSION}")

# Initialize paths
HOME_DIR = Path('/Volumes/ssd/01-ckj-postdoc')
DATA_DIR = HOME_DIR / 'LLM-alignment-data-generation'
PUBLIC_DATA_DIR = DATA_DIR / 'public-data' / 'open-australian-legal-qa'
GENERATED_DATA_DIR = DATA_DIR / 'generated-data' / 'open-australian-legal-qa'

# Check if directories exist
if not PUBLIC_DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {PUBLIC_DATA_DIR}")
if not GENERATED_DATA_DIR.exists():
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Check if input file exists
qa_file = PUBLIC_DATA_DIR / 'qa.jsonl'
if not qa_file.exists():
    raise FileNotFoundError(f"QA file not found: {qa_file}")

def load_api_key(api_key_path: str) -> str:
    """Load API key from file."""
    with open(api_key_path, 'r') as f:
        return f.read().strip()

class LegalResponseGenerator:
    def __init__(self, api_key_path: str):
        """Initialize the generator with path to API key file."""
        api_key = load_api_key(api_key_path)
        openai.api_key = api_key

        self.rules = {
            "pronoun": {
                "name": "Personal Pronoun Rule",
                "accepted": "Use personal pronouns, including inclusive 'we'",
                "rejected": "Avoid using any personal pronouns"
            },
            "voice": {
                "name": "Voice Rule",
                "accepted": "Use active voice",
                "rejected": "Use passive voice"
            },
            "tense": {
                "name": "Tense Rule",
                "accepted": "Use present tense",
                "rejected": "Use past tense"
            },
            "mood": {
                "name": "Mood Rule",
                "accepted": "Use polite imperative mood",
                "rejected": "Use imperative mood"
            },
            "words": {
                "name": "Emotional Words Rule",
                "accepted": "Use emotional words",
                "rejected": "Avoid using emotional words"
            }
        }

        self.emotion_lexicon = []
        
        # Define control settings for each feature
        self.features = {
            "tense": {"control_value": "present"},
            "pronoun": {"control_value": "personal"},
            "voice": {"control_value": "active"},
            "mood": {"control_value": "polite"},
            "words": {"control_value": False}
        }

    def analyze_data(self, file_path: str) -> Dict:
        """Analyze input data and return statistics."""
        questions = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data['question'])
        
        total_samples = len(questions)
        unique_samples = len(set(questions))
        
        stats = {
            'total_questions': total_samples,
            'unique_questions': unique_samples,
            'duplicates': total_samples - unique_samples,
            'train_size': int(total_samples * 0.8),
            'test_size': int(total_samples * 0.2)
        }
        
        print("\nDataset Statistics:")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Train set size (80%): {stats['train_size']}")
        print(f"Test set size (20%): {stats['test_size']}")
        
        return stats, questions

    def load_emotion_lexicon(self, filepath: str):
        """Load emotion words lexicon"""
        try:
            df = pd.read_csv(filepath)
            self.emotion_lexicon = df['lemma'].dropna().tolist()
            return True
        except Exception as e:
            print(f"Error loading emotion lexicon: {e}")
            return False

    def generate_prompt(self, question: str, test_feature: str, is_accepted: bool, control_settings: dict) -> str:
            """Generate a prompt that emphasizes maintaining the same meaning while controlling linguistic features."""
            prompt = f"""As a legal assistant, provide a response to the following legal question.

            Your response must follow these requirements:
            1. MOST IMPORTANT: The factual content and legal meaning of your response must be clear and complete
            2. Keep these linguistic features controlled:"""
            
            # Add control settings
            for feature, setting in control_settings.items():
                if feature != test_feature:
                    prompt += f"\n        - {feature}: {setting}"
                    
            # Add test feature instruction
            prompt += "\n        3. Follow this specific style requirement:"
            if test_feature == "words":
                prompt += f"\n        - {'USE emotional words' if is_accepted else 'AVOID emotional words'}"
            else:
                value = "accepted" if is_accepted else "rejected"
                prompt += f"\n        - For {test_feature}: Use {value} style"
                
            prompt += """

            Critical Instructions:
            - Your response should convey EXACTLY THE SAME legal information and meaning as you would normally provide
            - ONLY the linguistic style should change, not the underlying meaning or legal content
            - Keep the response concise and limited to a single sentence
            - Focus on answering the legal question while maintaining the required linguistic style"""
            
            prompt += f"\n\nQuestion: {question}"
            return prompt

    def validate_features(self, text: str, test_feature: str, is_accepted: bool, control_settings: dict) -> bool:
        """Validate text meets all feature requirements."""
        text_lower = text.lower()
        
        # Check emotional words if that's the test feature
        if test_feature == "words":
            has_emotion = any(word in text_lower.split() for word in self.emotion_lexicon)
            return has_emotion == is_accepted
        
        # Add validation for other features as needed
        return True

    def get_gpt4_response(self, prompt: str, test_feature: str, is_accepted: bool, control_settings: dict, max_attempts: int = 3) -> str:
        """Get response with controlled features."""
        for attempt in range(max_attempts):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {
                            "role": "system", 
                            "content": 
                            "You are a legal assistant specializing in Australian law cases."
                            },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7 + (attempt * 0.1)
                )
                
                response_text = response.choices[0].message['content'].strip()
                
                if self.validate_features(response_text, test_feature, is_accepted, control_settings):
                    return response_text
                    
                time.sleep(1)
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
                
        return ""

    def load_questions(self, file_path: str) -> List[str]:
        """Load questions from JSONL file."""
        questions = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data['question'])
        return questions

    def generate_dataset(self, questions: List[str], test_feature: str) -> pd.DataFrame:
        """Generate pairs of responses for a specific feature."""
        data = []
        control_settings = {k: v["control_value"] for k, v in self.features.items()}
        
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)} for {test_feature}")
            
            accepted_response = self.get_gpt4_response(
                self.generate_prompt(question, test_feature, True, control_settings),
                test_feature, True, control_settings
            )
            
            rejected_response = self.get_gpt4_response(
                self.generate_prompt(question, test_feature, False, control_settings),
                test_feature, False, control_settings
            )
            
            data.append({
                'question': question,
                'accepted_response': accepted_response,
                'rejected_response': rejected_response,
                'feature': test_feature
            })
            
        return pd.DataFrame(data)

    def process_all_features(self, data_path: str, output_dir: str, sample_size: int = None, batch_size: int = 5):  
        """Process all features and save results in batches."""
        questions = self.load_questions(data_path)
        
        if sample_size:
            questions = questions[:sample_size]
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for feature in self.features.keys():
            feature_dir = output_dir / feature
            feature_dir.mkdir(parents=True, exist_ok=True)
            
            for batch_idx in range(0, len(questions), batch_size):
                batch_questions = questions[batch_idx:batch_idx + batch_size]
                batch_num = batch_idx // batch_size + 1
                
                print(f"\nProcessing {feature} - Batch {batch_num}/{len(questions)//batch_size + 1}")
                
                df = self.generate_dataset(batch_questions, feature)
                
                # Save batch
                output_path = feature_dir / f'batch_{batch_num:03d}.csv'
                df.to_csv(output_path, index=False)
                print(f"Saved batch to {output_path}")
                
                # Save progress marker
                with open(feature_dir / 'progress.txt', 'w') as f:
                    f.write(f"Completed: {batch_num * batch_size}/{len(questions)}")
                
                time.sleep(1)

def create_dataset_card(stats: Dict) -> str:
    """Create the dataset card (README.md) for HuggingFace."""
    return f"""---
language:
- en
license: mit
task_categories:
- text-generation
task_ids:
- conditional-text-generation
---

# Empathetic Legal Responses Dataset

## Dataset Statistics
- Total questions in original data: {stats['total_questions']}
- Train set size (80%): {stats['train_size']} samples per rule
- Test set size (20%): {stats['test_size']} samples per rule

## Structure
```
empathetic-legal-responses/
├── README.md
└── data/
    ├── train/
    │   ├── pronoun.csv  ({stats['train_size']} samples)
    │   ├── voice.csv    ({stats['train_size']} samples)
    │   ├── tense.csv    ({stats['train_size']} samples)
    │   ├── mood.csv     ({stats['train_size']} samples)
    │   └── words.csv    ({stats['train_size']} samples)
    └── test/
        ├── pronoun.csv  ({stats['test_size']} samples)
        ├── voice.csv    ({stats['test_size']} samples)
        ├── tense.csv    ({stats['test_size']} samples)
        ├── mood.csv     ({stats['test_size']} samples)
        └── words.csv    ({stats['test_size']} samples)
```

## Rules Description
1. Pronoun Rule
- Accepted: Use personal pronouns, including inclusive "we"
- Rejected: Avoid using any personal pronouns

2. Voice Rule
- Accepted: Use active voice
- Rejected: Use passive voice

3. Tense Rule
- Accepted: Use present tense
- Rejected: Use past tense

4. Mood Rule
- Accepted: Use polite imperative mood
- Rejected: Use imperative mood

5. Words Rule
- Accepted: Use emotional words
- Rejected: Avoid using emotional words

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("cheonkamjeong/empathetic-legal-responses")
```

Each CSV file contains:
- question: Original legal question
- accepted_response: Response following the accepted style
- rejected_response: Response following the rejected style
- rule: The linguistic rule being applied
"""

def process_and_upload_dataset(input_file: str, openai_key: str, hf_token: str):
    """Process dataset and upload to HuggingFace with proper structure."""
    # Initialize generator
    generator = LegalResponseGenerator(openai_key)
    
    # Analyze data
    stats, questions = generator.analyze_data(input_file)
    
    # Create temporary directory for data
    temp_dir = Path("temp_data")
    (temp_dir / "train").mkdir(parents=True, exist_ok=True)
    (temp_dir / "test").mkdir(parents=True, exist_ok=True)
    
    # Process each rule
    for rule in tqdm(generator.rules.keys(), desc="Processing rules"):
        # Generate responses for all questions
        data = []
        for question in tqdm(questions, desc=f"Generating {rule} responses"):
            accepted = generator.get_gpt4_response(question, rule, True)
            rejected = generator.get_gpt4_response(question, rule, False)
            
            data.append({
                'question': question,
                'accepted_response': accepted,
                'rejected_response': rejected,
                'rule': rule
            })
        
        # Create DataFrame and split
        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Save splits
        train_df.to_csv(temp_dir / "train" / f"{rule}.csv", index=False)
        test_df.to_csv(temp_dir / "test" / f"{rule}.csv", index=False)
        
        print(f"\n{rule} split sizes:")
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Upload to HuggingFace
    api = HfApi()
    repo_id = "cheonkamjeong/empathetic-legal-responses"
    
    print("\nUploading to HuggingFace...")
    
    # Create dataset card
    readme_content = create_dataset_card(stats)
    
    # Upload files
    try:
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token
        )
        
        # Upload data files
        for split in ['train', 'test']:
            for rule in generator.rules.keys():
                file_path = temp_dir / split / f"{rule}.csv"
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"data/{split}/{rule}.csv",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token
                )
        
        print("Upload completed successfully!")
        
    except Exception as e:
        print(f"Error during upload: {e}")
    
    finally:
        # Cleanup
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Read API keys
    with open('openai_key.txt', 'r') as f:
        openai_key = f.read().strip()
    
    with open('hf_token.txt', 'r') as f:
        hf_token = f.read().strip()
    
    # Process and upload dataset
    process_and_upload_dataset(
        input_file="qa.jsonl",
        openai_key=openai_key,
        hf_token=hf_token
    )
