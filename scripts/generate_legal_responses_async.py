import asyncio
import aiohttp
import json
import logging
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache
from tqdm import tqdm
from huggingface_hub import HfApi
from itertools import islice
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        return cls(
            home_dir=home_dir,
            data_dir=data_dir,
            public_data_dir=public_data_dir,
            generated_data_dir=generated_data_dir,
            qa_file=qa_file,
            lexicon_file=lexicon_file
        )
    
@dataclass
class Rule:
    """Data class for rules."""
    name: str
    accepted: str
    rejected: str
    control_value: str

@dataclass
class DatasetStats:
    """Data class for dataset statistics."""
    total_questions: int
    train_size: int
    test_size: int

    def print_stats(self):
        """Print dataset statistics."""
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

class BatchProcessor:
    def __init__(self, batch_size: int = 5, max_concurrent_batches: int = 4):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        self.cache_dir = Path("response_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.progress_file = self.cache_dir / "progress.json"
        self.progress = self._load_progress()
    
    def batched(self, iterable, n):
        """Create batches from an iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    def validate_response(self, result: Dict[str, Any]) -> bool:
        """Simple validation for response quality."""
        text = result['response']
        return bool(text and text.strip())
    
    def _load_case(self, case: str, rule: str, is_accepted: bool) -> Optional[Dict[str, Any]]:
        """Load a case from cache if it exists."""
        cache_key = f"{hash(case)}_{rule}_{is_accepted}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache file: {e}")
        return None

    def _load_progress(self) -> Dict[str, Dict[str, bool]]:
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
    
    def _is_case_completed(self, case: str, rule: str, is_accepted: bool) -> bool:
        """Check if a specific case has been completed."""
        cache_key = f"{hash(case)}_{rule}_{is_accepted}"
        return self.progress.get(cache_key, False)

    async def process_case(
        self,
        case: str,
        rule: str,
        is_accepted: bool,
        generator: 'LegalResponseGenerator'
    ) -> Dict[str, Any]:
        """Process a single case with progress tracking."""
        cache_key = f"{hash(case)}_{rule}_{is_accepted}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if self._is_case_completed(cache_key, rule, is_accepted):
            with open(cache_file) as f:
                return json.load(f)
        
        try:
            response = await generator.get_gpt4_response_async(case, rule, is_accepted)
            if not response:
                raise ValueError("Empty response received")
                
            result = {
                'question': case,
                'response': response,
                'rule': rule,
                'is_accepted': is_accepted
            }
            
            if self.validate_response(result):
                with open(cache_file, 'w') as f:
                    json.dump(result, f)

                self.progress[cache_key] = True
                self._save_progress()
                
                return result
            else:
                raise ValueError("Response validation failed")
                
        except Exception as e:
            logger.error(f"Error processing case: {e}")
            return None

class LegalResponseGenerator:
    """Generate legal responses with controlled linguistic styles."""
    
    def __init__(self, api_key: str, lexicon_path: Path):
        self.api_key = api_key
        self.lexicon = Lexicon.from_csv(lexicon_path)
        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize linguistic rules with clear instructions."""
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
                rejected="Use passive voicet",
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

    async def get_gpt4_response_async(
        self,
        question: str,
        rule: str,
        is_accepted: bool,
        max_retries: int = 3
    ) -> Optional[str]:
        """Get response from GPT-4 asynchronously."""
        prompt = self.generate_prompt(question, rule, is_accepted)
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4-0125-preview",
                            "messages": [
                                {
                                    "role": "system", 
                                    "content": "You are a legal assistant specializing in Australian law cases. Your task is to provide clear, accurate legal information while following specific linguistic style requirements."
                                },
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.7
                        }
                    ) as response:
                        if response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', 60))
                            await asyncio.sleep(retry_after)
                            continue
                            
                        response_json = await response.json()
                        return response_json['choices'][0]['message']['content'].strip()
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return None

class DatasetUploader:
    """Handle dataset preparation and upload to HuggingFace."""
    
    def __init__(self, hf_token: str, repo_id: str):
        self.api = HfApi()
        self.hf_token = hf_token
        self.repo_id = repo_id

    async def generate_responses(
        self,
        temp_dir: Path,
        generator: LegalResponseGenerator,
        questions: List[str],
        batch_processor: BatchProcessor
    ):
        """Generate responses with progress tracking."""
        all_responses = []
        
        completed = sum(1 for k, v in batch_processor.progress.items() if v)
        logger.info(f"Resuming from {completed}/{len(questions)} unique questions")
        
        for rule in generator.rules:
            for batch in tqdm(batch_processor.batched(questions, batch_processor.batch_size),
                            desc=f"Processing {rule}"):
                accepted_tasks = []
                rejected_tasks = []
                
                for q in batch:
                    if not batch_processor._is_case_completed(q, rule, True):
                        accepted_tasks.append(
                            batch_processor.process_case(q, rule, True, generator)
                        )
                    if not batch_processor._is_case_completed(q, rule, False):
                        rejected_tasks.append(
                            batch_processor.process_case(q, rule, False, generator)
                        )
                
                if accepted_tasks:
                    accepted_results = await asyncio.gather(*accepted_tasks)
                if rejected_tasks:
                    rejected_results = await asyncio.gather(*rejected_tasks)
                
                for q in batch:
                    acc = batch_processor._load_case(q, rule, True)
                    rej = batch_processor._load_case(q, rule, False)
                    if acc and rej:
                        response_pair = {
                            'question': acc['question'],
                            'accepted_response': acc['response'],
                            'rejected_response': rej['response'],
                            'rule': rule
                        }
                        all_responses.append(response_pair)

        df = pd.DataFrame(all_responses)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_df.to_csv(temp_dir / "train.csv", index=False)
        test_df.to_csv(temp_dir / "test.csv", index=False)
        
        return DatasetStats(
            total_questions=len(df),
            train_size=len(train_df),
            test_size=len(test_df)
        )

    def create_dataset_card(self, stats: DatasetStats) -> str:
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

### Dataset Structure

The dataset consists of two CSV files:
- `train.csv` ({stats.train_size} samples)
- `test.csv` ({stats.test_size} samples)

Each CSV contains the following columns:
- `question`: Original legal case question
- `accepted_response`: Response following the accepted style
- `rejected_response`: Response following the rejected style
- `rule`: The linguistic rule being applied

### Data Fields

| Column | Type | Description |
|--------|------|-------------|
| question | string | The original legal case question |
| accepted_response | string | Response following the accepted style |
| rejected_response | string | Response following the rejected style |
| rule | string | The linguistic rule being applied |

### How to Use

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("cheonkamjeong/empathetic-legal-responses")

# Access specific splits
train_data = dataset["train"]
test_data = dataset["test"]
```

### Dataset Statistics
- Total samples: {stats.total_questions}
- Train set size (80%): {stats.train_size}
- Test set size (20%): {stats.test_size}
"""

    async def upload_dataset(self, temp_dir: Path, stats: DatasetStats):
        """Upload dataset to HuggingFace."""
        try:
            # Upload README
            readme_content = self.create_dataset_card(stats)
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

        except Exception as e:
            logger.error(f"Error during upload: {e}")
            raise

async def main():
    """Main execution function."""
    try:
        # Load API keys
        with open('api_key.txt', 'r') as f:
            openai_key = f.read().strip()
        with open('hf_token.txt', 'r') as f:
            hf_token = f.read().strip()

        # Initialize paths
        paths = PathConfig.create_default()
        logger.info(f"Using data directory: {paths.data_dir}")

        # Initialize components
        generator = LegalResponseGenerator(openai_key, paths.lexicon_file)
        batch_processor = BatchProcessor()
        uploader = DatasetUploader(
            hf_token=hf_token,
            repo_id="cheonkamjeong/empathetic-legal-responses"
        )

        # Load questions from JSONL file
        with open(paths.qa_file) as f:
            all_questions = [json.loads(line)['question'] for line in f]
            questions = list(set(all_questions))
            
        # Print initial stats
        logger.info(f"Loaded {len(questions)} questions")
        logger.info(f"Processing {len(questions)} unique questions")
        
        # Optional: limit questions for testing
        test_mode = False
        if test_mode:
            questions = questions[:100]
            logger.info(f"Test mode: using {len(questions)} questions")

        # Create temporary directory for data
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Generate and save responses
            stats = await uploader.generate_responses(
                temp_dir, generator, questions, batch_processor
            )
            
            # Print generation stats
            stats.print_stats()
            
            # Upload to HuggingFace
            await uploader.upload_dataset(temp_dir, stats)
            
            logger.info("Dataset generation and upload completed successfully!")

        finally:
            # Cleanup
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Temporary directory cleaned up")

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSONL file: {e}")
        raise

    except Exception as e:
        import traceback
        logger.error(f"Unexpected error in main execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())