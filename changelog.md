# Change Log

## Jan 14 Update by CKJ
- **Generated Responses**: 20 sample questions created.
- **Condition**: Each pair of responses (accepted vs. rejected) maintains the same propositional meaning.
- **Response Format**: Single-sentence responses limited to a maximum of 150 tokens.
- [**Models Tested**](https://github.com/ninackjeong/LLM-alignment-data-generation/blob/main/scripts/gpt-family.ipynb):
  - [GPT-4 (gpt-4)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/GPT-4/test-phase2)
  - [InstructGPT (gpt-3.5-turbo)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/InstructGPT/test-phase1)
  - [GPT-4o (gpt-4-0125-preview, latest version)](https://github.com/ninackjeong/LLM-alignment-data-generation/tree/main/generated-data/open-australian-legal-qa/GPT-4o/test-phase1).
- **Discussion**:
  - Imperatives seem unnatural as responses to wh-questions.
  - Granted, if we decide to use this, control for imperatives vs. polite imperatives.
- **Cost**:
  - Utilized my free credits so far.
- **Additional Testing**:
  - Claude 3.5 Sonnet test?
  - If sample responses above are fine, will try prompting with instances.
