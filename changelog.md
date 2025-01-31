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
- [**Models Tested**](https://github.com/ninackjeong/LLM-alignment-data-generation/blob/main/scripts/counsel-gpt-family.ipynb):
  - [GPT-4 (gpt-4)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/counsel-chat/GPT-4/test-phase1)

## Discussion items (updated on Jan 24)
- **Discussion**:
  - Imperatives seem unnatural as responses to wh-questions.
  - Upload data to our huggingface? Then, I will format them accordingly.
- **Cost**:
  - Utilized my free credits so far.
- **Additional Testing**:
  - Claude 3.5 Sonnet test? Just select one.
  - If sample responses above are fine, will use WordNet-Affect and Subtlex-US for explanability of lexical use.
  - Then, will try prompting with instances.
- **TODOs**:
  - 2,000 (4 rules) --> 25,000 (5 rules): Does this make model performance better?
  - Connect lexicon
  
## Jan 31 Update by CKJ
- **Response Generation with Examples**: Not better than those without examples. Actually, at some points, they are worse, so we will keep prompts without examples.
- **Lexical Processing**
