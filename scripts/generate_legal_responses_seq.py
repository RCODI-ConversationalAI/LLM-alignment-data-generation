from dataclasses import dataclass
import openai
import pandas as pd
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from tqdm import tqdm
import pkg_resources
import logging
from functools import lru_cache
from itertools import islice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Rule:
    """Data class for linguistic stylistic rules."""
    name: str
    accepted: str
    rejected: str
    control_value: str

@dataclass
class PathConfig:
    """Data class for path configuration."""
    home_dir: Path
    data_dir: Path
    public_data_dir: Path
    generated_data_dir: Path
    qa_file: Path
    lexicon_file: Path

    @classmethod
    def create_default(cls) -> 'PathConfig':
        """Create default path configuration."""
        home_dir = Path('/Volumes/ssd/01-ckj-postdoc')
        data_dir = home_dir / 'LLM-alignment-data-generation'
        public_data_dir = data_dir / 'public-data' / 'open-australian-legal-qa'
        generated_data_dir = data_dir / 'generated-data' / 'open-australian-legal-qa'
        qa_file = public_data_dir / 'qa.jsonl'
        lexicon_file = data_dir / 'lexicon' / 'affective_words_high_freq.csv'

        # Verify paths exist
        if not home_dir.exists():
            raise FileNotFoundError(f"Home directory not found: {home_dir}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not public_data_dir.exists():
            raise FileNotFoundError(f"Public data directory not found: {public_data_dir}")
        if not lexicon_file.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_file}")

        # Create generated data directory if it doesn't exist
        generated_data_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Public data directory: {public_data_dir}")
        logger.info(f"QA file: {qa_file}")
        logger.info(f"Lexicon file: {lexicon_file}")

        return cls(
            home_dir=home_dir,
            data_dir=data_dir,
            public_data_dir=public_data_dir,
            generated_data_dir=generated_data_dir,
            qa_file=qa_file,
            lexicon_file=lexicon_file
        )

class BatchProcessor:
    """Handle batch processing of questions with caching."""
    def __init__(self, batch_size: int = 5, cache_dir: Optional[Path] = None):
        self.batch_size = batch_size
        self.cache_dir = cache_dir or Path("response_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.progress_file = self.cache_dir / "progress.json"
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict[str, bool]:
        """Load progress from file or initialize new progress tracker."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading progress file: {e}")
                return {}
        return {}
    
    def _save_progress(self):
        """Save current progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    def _get_cache_key(self, question: str, rule: str, is_accepted: bool) -> str:
        """Generate a unique cache key for a response."""
        return f"{hash(question)}_{rule}_{is_accepted}"
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if a response is already cached."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        return cache_file.exists()
    
    def _save_to_cache(self, cache_key: str, data: dict):
        """Save response data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        self.progress[cache_key] = True
        self._save_progress()
    
    def _load_from_cache(self, cache_key: str) -> Optional[dict]:
        """Load response data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def process_questions(
        self,
        questions: List[str],
        generator: 'LegalResponseGenerator'
    ) -> List[dict]:
        """Process questions in batches with caching and resumption support."""
        all_responses = []
        total_questions = len(questions)
        
        # Calculate completion status
        completed_pairs = 0
        total_pairs = total_questions * len(generator.rules)
        
        for question in questions:
            for rule in generator.rules:
                acc_key = self._get_cache_key(question, rule, True)
                rej_key = self._get_cache_key(question, rule, False)
                if self._is_cached(acc_key) and self._is_cached(rej_key):
                    completed_pairs += 1
        
        logger.info("=== Generation Progress ===")
        logger.info(f"Total questions: {total_questions}")
        logger.info(f"Total pairs to generate: {total_pairs}")
        logger.info(f"Already completed: {completed_pairs} pairs")
        logger.info(f"Remaining: {total_pairs - completed_pairs} pairs")
        logger.info("=========================")
        
        for rule in tqdm(generator.rules, desc="Processing rules"):
            logger.info(f"Processing rule: {rule}")
            batches = list(self._create_batches(questions, self.batch_size))
            
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing {rule}")):
                batch_responses = []
                
                for question in batch:
                    accepted_key = self._get_cache_key(question, rule, True)
                    rejected_key = self._get_cache_key(question, rule, False)
                    
                    # Try to load from cache first
                    accepted_response = self._load_from_cache(accepted_key)
                    rejected_response = self._load_from_cache(rejected_key)
                    
                    # Generate missing responses
                    if not accepted_response:
                        response = generator.get_gpt4_response(question, rule, True)
                        if response:
                            accepted_response = {
                                'question': question,
                                'response': response,
                                'rule': rule,
                                'is_accepted': True
                            }
                            self._save_to_cache(accepted_key, accepted_response)
                    
                    if not rejected_response:
                        response = generator.get_gpt4_response(question, rule, False)
                        if response:
                            rejected_response = {
                                'question': question,
                                'response': response,
                                'rule': rule,
                                'is_accepted': False
                            }
                            self._save_to_cache(rejected_key, rejected_response)
                    
                    # If we have both responses, add to results
                    if accepted_response and rejected_response:
                        batch_responses.append({
                            'question': question,
                            'accepted_response': accepted_response['response'],
                            'rejected_response': rejected_response['response'],
                            'rule': rule
                        })
                
                # Add batch responses to overall results
                all_responses.extend(batch_responses)
                
                # Sleep between batches to avoid rate limits
                if batch_idx < len(batches) - 1:
                    time.sleep(2)
        
        return all_responses

    @staticmethod
    def _create_batches(iterable: List, n: int):
        """Create batches from an iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

@dataclass
class DatasetStats:
    """Data class for dataset statistics."""
    total_questions: int
    train_size: int
    test_size: int

    def print_stats(self):
        """Print dataset statistics."""
        logger.info("Dataset Statistics:")
        logger.info(f"Total questions: {self.total_questions}")
        logger.info(f"Train set size (80%): {self.train_size}")
        logger.info(f"Test set size (20%): {self.test_size}")

@dataclass
class Lexicon:
    """Data class for emotional word lexicon."""
    words: List[str]
    
    @classmethod
    def from_csv(cls, filepath: Path) -> 'Lexicon':
        """Load lexicon from CSV file."""
        try:
            df = pd.read_csv(filepath)
            words = df['word'].dropna().tolist()
            logger.info(f"Loaded {len(words)} words from lexicon")
            return cls(words=words)
        except Exception as e:
            logger.error(f"Error loading lexicon from {filepath}: {e}")
            raise

class ResponseValidator:
    """Validate responses against symbolic rules."""
    def __init__(self, lexicon: Optional[Lexicon] = None):
        self.lexicon = lexicon

    def validate_response(self, text: str, rule: str, is_accepted: bool) -> bool:
        """Validate response quality."""
        if not text or not text.strip():
            return False
            
        if rule == "words" and self.lexicon:
            has_emotion = any(word in text.lower() for word in self.lexicon.words)
            return has_emotion == is_accepted
            
        return True  # Basic validation for other rules

class LegalResponseGenerator:
    """Generate legal responses with controlled linguistic styles."""
    
    def __init__(self, api_key: str, lexicon_path: Path):
        self.api_key = api_key
        openai.api_key = api_key
        self.lexicon = Lexicon.from_csv(lexicon_path)
        self.validator = ResponseValidator(self.lexicon)
        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize linguistic rules."""
        self.rules = {
            "pronoun": Rule(
                name="Personal Pronoun Rule",
                accepted="Use personal pronouns including inclusive 'we'",
                rejected="Avoid personal pronouns",
                control_value="personal"
            ),
            "voice": Rule(
                name="Voice Rule",
                accepted="Use active voice",
                rejected="Use passive voice",
                control_value="active"
            ),
            "tense": Rule(
                name="Tense Rule",
                accepted="Use present tense",
                rejected="Use past tense",
                control_value="present"
            ),
            "mood": Rule(
                name="Mood Rule",
                accepted="Use polite imperative mood",
                rejected="Use imperative mood",
                control_value="polite"
            ),
            "words": Rule(
                name="Emotional Words Rule",
                accepted="Use emotional words",
                rejected="Avoid using emotional words",
                control_value="False"
            )
        }

    def generate_prompt(self, question: str, test_feature: str, is_accepted: bool) -> str:
        """Generate a prompt that emphasizes maintaining the same meaning while controlling linguistic features."""
        # Get control settings for all rules
        control_settings = {r: self.rules[r].control_value for r in self.rules}
        
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

    def get_gpt4_response(
        self, 
        question: str, 
        rule: str, 
        is_accepted: bool, 
        max_retries: int = 3
    ) -> Optional[str]:
        """Get response from GPT-4 with retry logic."""
        prompt = self.generate_prompt(question, rule, is_accepted)
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a legal assistant specializing in Australian law cases."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                response_text = response.choices[0].message['content'].strip()
                
                if self.validator.validate_response(response_text, rule, is_accepted):
                    return response_text
                    
                logger.warning(f"Response failed validation for {rule} rule")
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None

class DatasetUploader:
    """Handle dataset preparation and upload to HuggingFace."""
    
    def __init__(self, hf_token: str, repo_id: str):
        self.api = HfApi()
        self.hf_token = hf_token
        self.repo_id = repo_id

    def prepare_and_upload(
        self,
        responses: List[dict],
        stats: DatasetStats
    ):
        """Prepare and upload dataset to HuggingFace."""
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Convert responses to DataFrame and split
            df = pd.DataFrame(responses)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save splits
            train_df.to_csv(temp_dir / "train.csv", index=False)
            test_df.to_csv(temp_dir / "test.csv", index=False)
            
            # Create and upload README
            readme_content = self._create_dataset_card(stats)
            self.api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token
            )

            # Upload data files
            for split in ['train.csv', 'test.csv']:
                file_path = temp_dir / split
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"data/{split}",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.hf_token
                )
                
            logger.info("Dataset uploaded successfully!")
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

    def _create_dataset_card(self, stats: DatasetStats) -> str:
        """Create dataset card content."""
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

## Dataset Description

### Dataset Summary
This dataset contains Australian legal case responses with controlled linguistic variations. Each response maintains identical legal meaning while varying linguistic features according to specific rules.

### Dataset Statistics
- Total questions: {stats.total_questions}
- Train set size (80%): {stats.train_size}
- Test set size (20%): {stats.test_size}

### Data Fields
Each CSV file contains:
- question: Original legal question
- accepted_response: Response following the accepted style
- rejected_response: Response following the rejected style
- rule: The linguistic rule being applied

### Rules Description
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
"""

def main():
    """Main execution function."""
    try:
        # Check OpenAI version
        openai_version = pkg_resources.get_distribution("openai").version
        if openai_version != "0.28.0":
            raise ImportError(f"Please install openai==0.28.0. Current version: {openai_version}")

        logger.info("Starting initialization...")
        
        # Load configurations
        paths = PathConfig.create_default()
        
        # Validate paths
        if not paths.public_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {paths.public_data_dir}")
        if not paths.qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {paths.qa_file}")
        if not paths.lexicon_file.exists():
            raise FileNotFoundError(f"Lexicon file not found: {paths.lexicon_file}")
        
        # Load API keys
        try:
            with open('api_key.txt', 'r') as f:
                openai_key = f.read().strip()
                if not openai_key:
                    raise ValueError("OpenAI API key is empty")
        except FileNotFoundError:
            raise FileNotFoundError("api_key.txt not found")
            
        try:
            with open('hf_token.txt', 'r') as f:
                hf_token = f.read().strip()
                if not hf_token:
                    raise ValueError("HuggingFace token is empty")
        except FileNotFoundError:
            raise FileNotFoundError("hf_token.txt not found")

        # Initialize components
        generator = LegalResponseGenerator(openai_key, paths.lexicon_file)
        batch_processor = BatchProcessor(batch_size=5)
        
        # Load questions
        with open(paths.qa_file) as f:
            questions = [json.loads(line)['question'] for line in f]
            unique_questions = list(set(questions))
        
        # Calculate statistics
        total_questions = len(unique_questions)
        stats = DatasetStats(
            total_questions=total_questions,
            train_size=int(total_questions * 0.8),
            test_size=int(total_questions * 0.2)
        )
        stats.print_stats()
        
        # Optional: Test mode with limited questions
        test_mode = True  # Set to False for full dataset
        if test_mode:
            logger.info("Running in test mode with limited questions")
            unique_questions = unique_questions[:100]
            stats.total_questions = len(unique_questions)
            stats.train_size = int(stats.total_questions * 0.8)
            stats.test_size = int(stats.total_questions * 0.2)
        
        # Process questions in batches
        logger.info("Starting batch processing...")
        all_responses = batch_processor.process_questions(unique_questions, generator)
        
        # Initialize uploader and upload dataset
        uploader = DatasetUploader(
            hf_token=hf_token,
            repo_id="cheonkamjeong/empathetic-legal-responses"
        )
        
        uploader.prepare_and_upload(all_responses, stats)
        logger.info("Dataset generation and upload completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()