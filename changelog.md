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

## Feb 5 Update by CKJ
- **Data Stats**: 2,124 questions from the legal dataset, 785 questions from the counseling dataset
- **Added Validation Function**: double-check whether the rule is applied or not. may be helpful for evaluation
- **Lexical Processing**: just used affective words from WordNet-Affect. will do finer-grained ones after testing
- **Tested and Uploaded to HF**: check the response quality here: [100 questions; 500 responses](https://huggingface.co/datasets/cheonkamjeong/empathetic-legal-responses-test)
- **Paper**: Dataset part developed
- **Methods**: Decided to use identical questions across all rules rather than different question sets in order to:
  - Enable direct comparison of rule effects on same questions
  - Support more controlled ablation studies
  - Allow analysis of rule interactions on identical content
  - Provide better experimental control for reward model training

## Feb 10-11 Update by CKJ
- **Batch Processing**: Implemented two versions – [**asynchronous processing**](https://github.com/RCODI-ConversationalAI/LLM-alignment-data-generation/blob/main/scripts/generate_legal_responses_async.py) and [**sequential batch execution**](https://github.com/RCODI-ConversationalAI/LLM-alignment-data-generation/blob/main/scripts/generate_legal_responses_seq.py) (encountered connection errors due to quota limits, so made a sequential version as well. If quota limits are fixed, will try asynchronous processing again. Estimated time: around 1 hour). Will apply the same process to the counseling data after completing this.
- **Lexical Processing** ([affective parser here](https://github.com/RCODI-ConversationalAI/LLM-alignment-data-generation/blob/main/scripts/affective_parser.py)): Integrated **WordNet-Affect** and **WordNet synsets**, incorporating **frequency data from SUBTLEX-US** to filter for **high-frequency items only**.
- **Issues**: Encountered **quota limitations** affecting API usage.
- **Workflow**: Running **UCI server tests** while generating data. Testing the **reward model** on the server. Conducting a **fine-tuning test** with **100 sample data**.
- **Your TODOs**: Check **quota limitations**. Conduct **Gilbreth server testing** with the sample data.
