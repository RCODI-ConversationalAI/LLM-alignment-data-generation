import asyncio
import aiohttp
import json
import logging
import pandas as pd
import random
import shutil
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache
from tqdm import tqdm
from huggingface_hub import HfApi
from itertools import islice
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
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
        # Get base directory from environment variable or use default
        data_dir = Path(os.getenv('DATA_DIR', '/home/cheonkaj/projects/LLM-alignment-data-generation'))
        
        # Define subdirectories
        public_data_dir = data_dir / 'public-data'
        generated_data_dir = data_dir / 'generated-data'
        qa_file = public_data_dir / 'open-australian-legal-qa' / 'qa.jsonl'
        lexicon_file = data_dir / 'lexicon' / 'affective_words_high_freq.csv'
        
        # Validate paths
        if not qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {qa_file}")
        if not lexicon_file.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_file}")
        
        logger.info(f"Using data directory: {data_dir}")
        logger.info(f"QA file: {qa_file}")
        logger.info(f"Lexicon file: {lexicon_file}")
        
        return cls(
            home_dir=data_dir.parent,
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
        self.backup_interval = 10
        self.processed_since_backup = 0
        self.progress = self._load_progress()
    
    def batched(self, iterable, n):
        """Create batches from an iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    def backup_progress(self):
        """Create a backup of the progress file."""
        try:
            backup_file = self.progress_file.with_suffix('.json.backup')
            shutil.copy2(self.progress_file, backup_file)
            logger.info(f"Progress backup created: {backup_file}")
            self.processed_since_backup = 0
        except Exception as e:
            logger.error(f"Failed to create progress backup: {e}")

    def _save_progress(self):
        """Save current progress to file with backup."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
        
        self.processed_since_backup += 1
        if self.processed_since_backup >= self.backup_interval:
            self.backup_progress()

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

    def _is_case_completed(self, cache_key: str, rule: str, is_accepted: bool) -> bool:
        """Check if a specific case has been completed."""
        return self.progress.get(cache_key, False)

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

    def validate_response(self, result: Dict[str, Any]) -> bool:
        """Validate response quality."""
        text = result.get('response', '')
        if not text or not isinstance(text, str):
            return False
        return len(text.strip()) >= 10

    async def process_case(
        self,
        case: str,
        rule: str,
        is_accepted: bool,
        generator: 'LegalResponseGenerator'
    ) -> Dict[str, Any]:
        """Process a single case with enhanced progress tracking."""
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
                'is_accepted': is_accepted,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            if self.validate_response(result):
                with open(cache_file, 'w') as f:
                    json.dump(result, f)

                self.progress[cache_key] = True
                self._save_progress()
                
                # Log progress for this rule
                completed = sum(1 for k, v in self.progress.items() if v and k.endswith(f"_{rule}_{'True' if is_accepted else 'False'}"))
                logger.info(f"Rule: {rule}, Type: {'accepted' if is_accepted else 'rejected'}, Completed: {completed}")
                
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
        """Generate a prompt that emphasizes maintaining the same meaning while controlling linguistic styles."""
        
        # Define specific examples for each rule style
        style_examples = {
            "pronoun": {
                "accepted": "Example: 'We recommend filing the claim' (using 'we', 'you')",
                "rejected": "Example: 'The claim should be filed' (no personal pronouns)",
            },
            "voice": {
                "accepted": "Example: 'The court requires these documents' (grammatical subject is the agent)",
                "rejected": "Example: 'These documents are required by the court' (grammatical subject is the patient/theme)",
            },
            "tense": {
                "accepted": "Example: 'The law requires...' (present tense)",
                "rejected": "Example: 'The law required...' (past tense)",
            },
            "mood": {
                "accepted": "Example: 'Please file the claim immediately' (polite suggestion)",
                "rejected": "Example: 'File the claim immediately' (direct command)",
            },
            "words": {
                "accepted": "Example: 'We understand this (concerning) situation requires...' (with emotional words)",
                "rejected": "Example: 'We know this situation requires...' (neutral language)",
            }
        }

        # Get example for current rule
        example = style_examples[test_feature]["accepted" if is_accepted else "rejected"]
            
        prompt = f"""As a legal assistant, provide a response to the following legal question.

        Your response must follow these requirements:
        1. MOST IMPORTANT: The factual content and legal meaning of your response must be clear and complete
        2. Follow this ONE specific style rule:"""
        
        # Add ONLY the test feature instruction
        if test_feature == "words":
            prompt += f"\n    - {'USE emotional words' if is_accepted else 'AVOID emotional words'}"
        else:
            prompt += f"\n    - {self.rules[test_feature].accepted if is_accepted else self.rules[test_feature].rejected}"
        
        # Add the example
        prompt += f"\n    {example}"
                
        prompt += """

        Critical Instructions:
        - Your response MUST convey EXACTLY THE SAME legal information as would be in a standard response
        - Change ONLY the linguistic style, never the facts or legal content
        - Keep your response to EXACTLY ONE SENTENCE - no more, no less
        - Make sure your response clearly follows the style rule while maintaining identical factual content
        - If referring to case names or legal sources, maintain exactly the same references in both styles"""
        
        prompt += f"\n\nQuestion: {question}"
        return prompt

    async def get_gpt4_response_async(
            self,
            question: str,
            rule: str,
            is_accepted: bool,
            max_retries: int = 3,
            max_tokens: int = 128
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
                                "max_tokens": max_tokens,
                                "temperature": 0.7
                            }
                        ) as response:
                            if response.status == 429:  # Rate limit
                                retry_after = int(response.headers.get('Retry-After', 60))
                                logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                                await asyncio.sleep(retry_after)
                                continue
                                
                            response_json = await response.json()
                            if 'error' in response_json:
                                logger.error(f"API error: {response_json['error']}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                return None
                                
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
        total_to_process = len(questions) * len(generator.rules) * 2  # Each question gets processed for each rule, both accepted and rejected
        logger.info(f"Resuming from {completed}/{total_to_process} total tasks ({(completed/total_to_process)*100:.1f}%)")
        
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
                
                # Process results and ensure no duplicates
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
                        # Check if we already have this pair
                        is_duplicate = False
                        for existing in all_responses:
                            if existing['question'] == response_pair['question'] and existing['rule'] == response_pair['rule']:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_responses.append(response_pair)

        # Create a DataFrame and remove any remaining duplicates
        df = pd.DataFrame(all_responses).drop_duplicates(subset=['question', 'rule'])
        
        # Split into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Save to CSV files
        train_df.to_csv(temp_dir / "train.csv", index=False)
        test_df.to_csv(temp_dir / "test.csv", index=False)
        
        # Save sample responses for quality check
        sample_size = min(10, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(temp_dir / "samples.csv", index=False)
        logger.info(f"Saved {sample_size} sample responses to {temp_dir}/samples.csv for quality check")
        
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
    """Main execution function with enhanced error handling and progress tracking."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get file paths from environment variables
        hf_token_path = os.getenv('HUGGINGFACE_TOKEN_PATH')
        api_key_path = os.getenv('OPENAI_API_KEY')
        
        if not hf_token_path or not api_key_path:
            raise ValueError("Required environment variables not found")
        
        # Read API keys from files
        try:
            with open(hf_token_path, 'r') as f:
                hf_token = f.read().strip()
            with open(api_key_path, 'r') as f:
                openai_key = f.read().strip()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find required secret file: {e.filename}")

        # Create directories for accumulated data
        main_data_dir = Path("accumulated_data4")
        main_data_dir.mkdir(exist_ok=True)
        
        # Initialize or load accumulated DataFrames
        try:
            if (main_data_dir / "train.csv").exists():
                accumulated_train_df = pd.read_csv(main_data_dir / "train.csv")
                accumulated_test_df = pd.read_csv(main_data_dir / "test.csv")
                logger.info(f"Loaded existing data: {len(accumulated_train_df)} train, {len(accumulated_test_df)} test samples")
            else:
                accumulated_train_df = pd.DataFrame()
                accumulated_test_df = pd.DataFrame()
                logger.info("Starting with new empty dataset")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            accumulated_train_df = pd.DataFrame()
            accumulated_test_df = pd.DataFrame()

        # Initialize paths and components
        paths = PathConfig.create_default()
        generator = LegalResponseGenerator(openai_key, paths.lexicon_file)
        uploader = DatasetUploader(
            hf_token=hf_token,
            repo_id="cheonkamjeong/empathetic-legal-responses"
        )

        # Get batch range from environment 
        START_BATCH = int(os.getenv('START_BATCH', 1))
        END_BATCH = int(os.getenv('END_BATCH', 1))
        BATCH_SIZE = 10  # Reduced for testing (original was 200)

        # Load all questions
        with open(paths.qa_file) as f:
            all_questions = [json.loads(line)['question'] for line in f]
            # Remove duplicates and ensure consistent order
            unique_questions = sorted(list(set(all_questions)))

        # Calculate batch ranges
        total_questions = len(unique_questions)
        start_idx = (START_BATCH - 1) * BATCH_SIZE
        end_idx = min(END_BATCH * BATCH_SIZE, total_questions)
        
        # Get questions for specified range
        questions_to_process = unique_questions[start_idx:end_idx]
        
        # Log batch processing information
        logger.info(f"Processing batches {START_BATCH} to {END_BATCH}")
        logger.info(f"Questions range: {start_idx+1} to {end_idx}")
        logger.info(f"Total questions in range: {len(questions_to_process)}")

        # Calculate total batches for this range
        total_batches = (len(questions_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

        # Process each batch in the specified range
        for batch_idx, batch_start in enumerate(range(0, len(questions_to_process), BATCH_SIZE), 1):
            batch_end = min(batch_start + BATCH_SIZE, len(questions_to_process))
            batch_questions = questions_to_process[batch_start:batch_end]
            
            logger.info(f"\nProcessing batch {batch_idx}/{total_batches}")
            logger.info(f"Batch size: {len(batch_questions)} questions")
            
            # Initialize batch processor for this batch
            batch_processor = BatchProcessor(batch_size=1, max_concurrent_batches=1)  # Process one question at a time for testing
            
            # Create temporary directory for this batch
            temp_dir = Path(f"temp_data_batch_{batch_idx}")
            temp_dir.mkdir(exist_ok=True)

            try:
                # Generate responses for this batch
                batch_stats = await uploader.generate_responses(
                    temp_dir, generator, batch_questions, batch_processor
                )
                
                logger.info(f"\nBatch {batch_idx} Statistics:")
                batch_stats.print_stats()

                # Read and accumulate results
                batch_train_df = pd.read_csv(temp_dir / "train.csv")
                batch_test_df = pd.read_csv(temp_dir / "test.csv")
                
                # Check for and remove duplicates before concatenating
                existing_train_questions = set(accumulated_train_df['question']) if not accumulated_train_df.empty else set()
                existing_test_questions = set(accumulated_test_df['question']) if not accumulated_test_df.empty else set()
                
                # Remove questions that are already in the accumulated dataset
                batch_train_df = batch_train_df.drop_duplicates(subset=['question', 'rule'])
                batch_test_df = batch_test_df.drop_duplicates(subset=['question', 'rule'])
                
                # Concatenate with accumulated data
                accumulated_train_df = pd.concat([accumulated_train_df, batch_train_df], ignore_index=True)
                accumulated_test_df = pd.concat([accumulated_test_df, batch_test_df], ignore_index=True)
                
                # Final deduplication 
                accumulated_train_df = accumulated_train_df.drop_duplicates(subset=['question', 'rule'])
                accumulated_test_df = accumulated_test_df.drop_duplicates(subset=['question', 'rule'])
                
                # Save accumulated results
                accumulated_train_df.to_csv(main_data_dir / "train.csv", index=False)
                accumulated_test_df.to_csv(main_data_dir / "test.csv", index=False)

                # Update total statistics
                total_stats = DatasetStats(
                    total_questions=len(accumulated_train_df) + len(accumulated_test_df),
                    train_size=len(accumulated_train_df),
                    test_size=len(accumulated_test_df)
                )
                
                # Upload accumulated results to HuggingFace
                await uploader.upload_dataset(main_data_dir, total_stats)
                
                logger.info(f"\nAccumulated Statistics after batch {batch_idx}:")
                total_stats.print_stats()

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                # Save progress before continuing
                accumulated_train_df.to_csv(main_data_dir / "train.csv", index=False)
                accumulated_test_df.to_csv(main_data_dir / "test.csv", index=False)
                continue
                
            finally:
                # Cleanup batch temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temporary directory for batch {batch_idx} cleaned up")

            # Add delay between batches
            if batch_idx < total_batches:
                delay = 60  # 1 minute delay
                logger.info(f"Waiting {delay} seconds before starting next batch...")
                await asyncio.sleep(delay)

        logger.info("\nAll batches completed!")
        logger.info(f"Final accumulated data saved in: {main_data_dir}")

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Save any accumulated data before raising the error
        if 'accumulated_train_df' in locals():
            accumulated_train_df.to_csv(main_data_dir / "train.csv", index=False)
            accumulated_test_df.to_csv(main_data_dir / "test.csv", index=False)
        raise

if __name__ == "__main__":
    asyncio.run(main())