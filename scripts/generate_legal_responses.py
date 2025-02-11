from dataclasses import dataclass
from enum import Enum, auto
import openai
import pandas as pd
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import pkg_resources
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Rule:
    """Data class for linguistic rules."""
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

    @classmethod
    def create_default(cls) -> 'PathConfig':
        """Create default path configuration."""
        home_dir = Path('/Volumes/ssd/01-ckj-postdoc')
        data_dir = home_dir / 'LLM-alignment-data-generation'
        public_data_dir = data_dir / 'public-data' / 'open-australian-legal-qa'
        generated_data_dir = data_dir / 'generated-data' / 'open-australian-legal-qa'
        qa_file = public_data_dir / 'qa.jsonl'
        return cls(home_dir, data_dir, public_data_dir, generated_data_dir, qa_file)

@dataclass
class DatasetStats:
    """Data class for dataset statistics."""
    total_questions: int
    unique_questions: int
    duplicates: int
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
    words: List[str]
    
    @classmethod
    def from_csv(cls, filepath: Path) -> 'Lexicon':
        """Load lexicon from CSV file."""
        try:
            df = pd.read_csv(filepath)
            words = df['lemma'].dropna().tolist()
            logger.info(f"Loaded {len(words)} words from lexicon")
            logger.debug(f"Sample words: {words[:10]}")
            return cls(words=words)
        except Exception as e:
            logger.error(f"Error loading lexicon from {filepath}: {e}")
            raise

    def contains_emotional_words(self, text: str) -> bool:
        """Check if text contains words from the emotion lexicon."""
        text_lower = text.lower()
        text_words = set(text_lower.split())

        matches = [word for word in self.words if word in text_words]
        if matches:
            logger.debug(f"Found emotional words: {matches}")
        return bool(matches)

class ResponseValidator:
    """Validate responses against symbolic rules."""
    def __init__(self, lexicon: Optional[Lexicon] = None):
        """Initialize validator with optional lexicon."""
        self.lexicon = lexicon

    def validate_response(
        self, 
        text: str, 
        rule: str, 
        is_accepted: bool, 
        control_settings: Dict[str, str]
    ) -> bool:
        """Validate and log responses but don't reject for testing."""
        if rule == "words" and self.lexicon:
            has_emotion = self.lexicon.contains_emotional_words(text)
            logger.info(f"\nRule: {rule}, Expected emotional words: {is_accepted}")
            logger.info(f"Response: {text}")
            logger.info(f"Has emotional words: {has_emotion}")
        return True  # Always return True for testing

class LegalResponseGenerator:
    def __init__(self, api_key: str, lexicon_path: Optional[Path] = None):
        """Initialize the generator with API key and rules."""
        self.api_key = api_key
        openai.api_key = api_key

        self.lexicon = Lexicon.from_csv(lexicon_path) if lexicon_path else None
        self.validator = ResponseValidator(self.lexicon)

        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize symbolic rules."""
        self.rules: Dict[str, Rule] = {
            "pronoun": Rule(
                name="Personal Pronoun Rule",
                accepted="Use personal pronouns, including inclusive 'we'",
                rejected="Avoid using any personal pronouns",
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

    @staticmethod
    @lru_cache(maxsize=1)
    def load_questions(file_path: Path) -> List[str]:
        """Load and cache questions from JSONL file."""
        try:
            with open(file_path) as f:
                return [json.loads(line)['question'] for line in f]
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise

    def analyze_data(self, file_path: Path) -> Tuple[DatasetStats, List[str]]:
        """Analyze input data and return statistics."""
        questions = self.load_questions(file_path)
        total_samples = len(questions)
        
        stats = DatasetStats(
            total_questions=total_samples,
            unique_questions=len(set(questions)),
            duplicates=total_samples - len(set(questions)),
            train_size=int(total_samples * 0.8),
            test_size=int(total_samples * 0.2)
        )
        
        stats.print_stats()
        return stats, questions

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

    def get_gpt4_response(
        self, 
        question: str, 
        rule: str, 
        is_accepted: bool, 
        max_retries: int = 3
    ) -> Optional[str]:
        """Get response from GPT-4 with retry logic and validation."""
        control_settings = {r: self.rules[r].control_value for r in self.rules}
        prompt = self.generate_prompt(question, rule, is_accepted, control_settings)
        
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "system", "content": "You are a legal assistant specializing in Australian law cases."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                response_text = response.choices[0].message['content'].strip()
                
                # Validate response
                if self.validator.validate_response(response_text, rule, is_accepted, control_settings):
                    return response_text
                
                logger.warning(f"Response failed validation for {rule} rule, retrying...")
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to get valid response after {max_retries} attempts")
        return None

class DatasetUploader:
    """Handle dataset preparation and upload to HuggingFace."""
    def __init__(self, hf_token: str, repo_id: str):
        self.api = HfApi()
        self.hf_token = hf_token
        self.repo_id = repo_id

    def prepare_and_upload(
        self, 
        generator: LegalResponseGenerator,
        questions: List[str],
        stats: DatasetStats
    ):
        """Prepare and upload dataset to HuggingFace."""
        temp_dir = Path("temp_data")
        try:
            self._prepare_directory_structure(temp_dir)
            self._generate_and_save_data(temp_dir, generator, questions)
            self._upload_to_huggingface(temp_dir, stats, generator.rules)
        finally:
            self._cleanup(temp_dir)

    def _prepare_directory_structure(self, temp_dir: Path):
        """Create temporary directory structure."""
        for split in ['train', 'test']:
            (temp_dir / split).mkdir(parents=True, exist_ok=True)

    def _generate_and_save_data(
        self, 
        temp_dir: Path,
        generator: LegalResponseGenerator,
        questions: List[str]
    ):
        """Generate and save data for each rule."""
        for rule in tqdm(generator.rules.keys(), desc="Processing rules"):
            data = []
            for question in tqdm(questions, desc=f"Generating {rule} responses"):
                accepted = generator.get_gpt4_response(question, rule, True)
                rejected = generator.get_gpt4_response(question, rule, False)
                
                if accepted and rejected:  # Only add if both responses were generated
                    data.append({
                        'question': question,
                        'accepted_response': accepted,
                        'rejected_response': rejected,
                        'rule': rule
                    })

            df = pd.DataFrame(data)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            train_df.to_csv(temp_dir / "train" / f"{rule}.csv", index=False)
            test_df.to_csv(temp_dir / "test" / f"{rule}.csv", index=False)

    def _upload_to_huggingface(
        self, 
        temp_dir: Path,
        stats: DatasetStats,
        rules: Dict[str, Rule]
    ):
        """Upload data to HuggingFace."""
        try:
            # Upload README
            readme_content = self._create_dataset_card(stats, rules)
            self.api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token
            )

            # Upload data files
            for split in ['train', 'test']:
                for rule in rules.keys():
                    file_path = temp_dir / split / f"{rule}.csv"
                    self.api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"data/{split}/{rule}.csv",
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.hf_token
                    )

            logger.info("Upload completed successfully!")

        except Exception as e:
            logger.error(f"Error during upload: {e}")
            raise

    @staticmethod
    def _cleanup(temp_dir: Path):
        """Clean up temporary directory."""
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)

    @staticmethod
    def _create_dataset_card(stats: DatasetStats, rules: Dict[str, Rule]) -> str:
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

## Dataset Statistics
- Total questions in original data: {stats.total_questions}
- Train set size (80%): {stats.train_size} samples per rule
- Test set size (20%): {stats.test_size} samples per rule

## Structure
```
empathetic-legal-responses/
├── README.md
└── data/
    ├── train/
    │   ├── pronoun.csv  ({stats.train_size} samples)
    │   ├── voice.csv    ({stats.train_size} samples)
    │   ├── tense.csv    ({stats.train_size} samples)
    │   ├── mood.csv     ({stats.train_size} samples)
    │   └── words.csv    ({stats.train_size} samples)
    └── test/
        ├── pronoun.csv  ({stats.test_size} samples)
        ├── voice.csv    ({stats.test_size} samples)
        ├── tense.csv    ({stats.test_size} samples)
        ├── mood.csv     ({stats.test_size} samples)
        └── words.csv    ({stats.test_size} samples)
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

def main():
    """Main execution function."""
    try:
        # Check OpenAI version
        openai_version = pkg_resources.get_distribution("openai").version
        if openai_version != "0.28.0":
            raise ImportError(f"Please install openai==0.28.0. Current version: {openai_version}")

        logger.info("Starting initialization...")
        
        # Load configurations and check paths
        paths = PathConfig.create_default()
        logger.info(f"Checking paths...")
        logger.info(f"Public data dir: {paths.public_data_dir}")
        logger.info(f"QA file: {paths.qa_file}")
        logger.info(f"Lexicon path: {paths.data_dir / 'lexicon' / 'affective_words_high_freq.csv'}")
        
        # Validate paths
        if not paths.public_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {paths.public_data_dir}")
        if not paths.qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {paths.qa_file}")

        # Check lexicon file
        lexicon_path = paths.data_dir / "lexicon" / "affective_words_high_freq.csv"
        if not lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
        
        # Check API keys
        logger.info("Checking API keys...")
        try:
            with open('api_key.txt', 'r') as f:
                openai_key = f.read().strip()
                if not openai_key:
                    raise ValueError("OpenAI API key is empty")
        except FileNotFoundError:
            raise FileNotFoundError("openai_key.txt not found")
            
        try:
            with open('hf_token.txt', 'r') as f:
                hf_token = f.read().strip()
                if not hf_token:
                    raise ValueError("HuggingFace token is empty")
        except FileNotFoundError:
            raise FileNotFoundError("hf_token.txt not found")

        # Initialize generator and test lexicon loading
        logger.info("Initializing generator and loading lexicon...")
        generator = LegalResponseGenerator(openai_key, lexicon_path=lexicon_path)
        
        # Test question loading
        logger.info("Testing question loading...")
        stats, questions = generator.analyze_data(paths.qa_file)
        logger.info(f"Successfully loaded {len(questions)} questions")
        
        # Only proceed if everything is loaded correctly
        user_input = input("\nAll components loaded successfully. Proceed with generation? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Operation cancelled by user")
            return

        test_mode = True  # Set to False for full dataset
        if test_mode:
            logger.info("Running in test mode with 30 questions")
            questions = questions[:100]
            stats.total_questions = len(questions)
            stats.train_size = int(stats.total_questions * 0.8)
            stats.test_size = int(stats.total_questions * 0.2)

        # Initialize uploader
        uploader = DatasetUploader(
            hf_token=hf_token,
            repo_id="cheonkamjeong/empathetic-legal-responses"
        )

        # Process and upload dataset
        uploader.prepare_and_upload(generator, questions, stats)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
