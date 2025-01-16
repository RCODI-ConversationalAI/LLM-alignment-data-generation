# Change Log

## Jan 14 Update by CKJ
- **Generated Responses with Legal Data**: 20 sample questions per rule created.
- **Condition**: Each pair of responses (accepted vs. rejected) maintains the same propositional meaning.
- **Response Format**: Single-sentence responses limited to a maximum of 150 tokens.
- [**Models Tested**](https://github.com/ninackjeong/LLM-alignment-data-generation/blob/main/scripts/gpt-family.ipynb):
  - [GPT-4 (gpt-4)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/GPT-4/test-phase2)
  - [InstructGPT (gpt-3.5-turbo)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/InstructGPT/test-phase1)
  - [GPT-4o (gpt-4-0125-preview, latest version)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/GPT-4o/test-phase1).

## Jan 15 Update by CKJ
- **Generated Responses with Counseling Data**: Duplicated questions (called 'history' in this dataset) removed. 5 sample questions per rule created.
- **Condition**: Same as above
- **Response Format**: Same as above
- [**Models Tested**](https://github.com/ninackjeong/LLM-alignment-data-generation/blob/main/scripts/gpt-family.ipynb):
  - [GPT-4 (gpt-4)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/GPT-4/test-phase2)
  - [InstructGPT (gpt-3.5-turbo)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/InstructGPT/test-phase1)
  - [GPT-4o (gpt-4-0125-preview, latest version)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/GPT-4o/test-phase1).

## Discussion items (updated on Jan 15)
- **Discussion**:
  - Imperatives seem unnatural as responses to wh-questions.
  - Granted, if we decide to use this, control for imperatives vs. polite imperatives; controlled for counseling data generation, will do it again with legal data.
- **Cost**:
  - Utilized my free credits so far.
- **Additional Testing**:
  - Claude 3.5 Sonnet test?
  - If sample responses above are fine, will use WordNet-Affect and Subtlex-US for explanability of lexical use.
  - Then, will try prompting with instances.
