---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license: other
size_categories:
- 1K<n<10K
source_datasets:
- umarbutler/open-australian-legal-corpus
task_categories:
- question-answering
- text-generation
- text2text-generation
task_ids:
- closed-domain-qa
pretty_name: Open Australian Legal QA
license_name: open-australian-legal-corpus
license_link: https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus/blob/main/LICENCE.md
tags:
- law
- legal
- australia
- question-answering
- qa
- question-answer
- text-generation
- llm
- chatbot
- conversational-ai
- generative-ai
- natural-language-understanding
- fine-tuning
language_details: en-AU, en-GB
viewer: true
dataset_info:
  config_name: train
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: text
    dtype: string
  - name: prompt
    dtype: string
  - name: source
    struct:
    - name: version_id
      dtype: string
    - name: type
      dtype: string
    - name: jurisdiction
      dtype: string
    - name: source
      dtype: string
    - name: citation
      dtype: string
    - name: url
      dtype: string
    - name: text
      dtype: string
  splits:
  - name: train
    num_bytes: 13243775
    num_examples: 2124
  download_size: 13538191
  dataset_size: 13243775
---
<!-- To update the above `dataset_info` section, please run the following command: `datasets-cli test open_australian_legal_qa.py --save_info --all_configs`. -->

# **Open Australian Legal QA ‍⚖️**
<a href="https://huggingface.co/datasets/umarbutler/open-australian-legal-qa" alt="Release"><img src="https://img.shields.io/badge/release-v2.0.0-green"></a>

Open Australian Legal QA is the first open dataset of Australian legal questions and answers.

Comprised of 2,124 questions and answers synthesised by `gpt-4` from the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus), the largest open database of Australian law, the dataset is intended to facilitate the development of legal AI assistants in Australia.

To ensure its accessibility to as wide an audience as possible, the dataset is distributed under the same licence as the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus/blob/main/LICENCE.md).

## Usage 👩‍💻
The below code snippet illustrates how the dataset may be loaded with the [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) Python library:
```python
from datasets import load_dataset

corpus = load_dataset('umarbutler/open-australian-legal-qa', split='train')
```

To speed up the loading of the dataset, you may wish to install [`orjson`](https://github.com/ijl/orjson).

## Structure 🗂️
The dataset is stored in [qa.jsonl](https://huggingface.co/datasets/umarbutler/open-australian-legal-qa/blob/main/qa.jsonl), a json lines file where each line represents a question-answer pair consisting of four keys:
| Key | Description |
| --- | --- |
| question | The text of the question. |
| answer | The text of the answer to the question. |
| text | The text of the question and answer in the format `Question: {question}\nAnswer: {answer}`. |
| prompt | The text of the prompt used to generate the question-answer pair. |
| source | A dictionary representing the document from which the question-answer pair was synthesised, sharing the same keys as documents in the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus), with the `text` field constituting the text of the chunk used to generate the pair. |

## Methodology 🧪
2,124 documents from the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) were randomly sampled, barring bills and documents consisting entirely of whitespace. These documents were then split into semantically meaningful chunks up to 384-tokens-long (as determined by [`tiktoken`](https://github.com/openai/tiktoken)'s tokeniser for `gpt-4`) with the [`semchunk`](https://github.com/umarbutler/semchunk) Python library. 

Chunks that consisted entirely of whitespace, contained 6 or more consecutive periods, ignoring whitespace (indicating that they contained a table of contents) or that were less than 96-tokens-long were discarded. A single chunk was randomly selected from each document (for those documents with a chunk to select) and subsequently cleaned of consecutive newlines, consecutive whitespace and lines consisting entirely of whitespace.

These chunks were then embedded into the following prompt, with the names of jurisdictions and types being capitalised and stripped of hyphens:
```xml
# Snippet
The snippet from an Australian legal document from which you must synthesise a question and answer is provided below.
<document_metadata>
<document_title><!-- insert citation here --></document_title>
<document_jurisdiction><!-- insert jurisdiction here --></document_jurisdiction>
<document_type><!-- insert type here --></document_type>
</document_metadata>
<snippet>
<!-- insert text here -->
</snippet>

# Format
You must format your response as follows:
<format>
# Question
{A question related to the snippet, or a topic discussed therein.}

# Answer
{The answer to the question, extracted from the snippet.}
</format>

# Instructions
You must act as a question-and-answer synthesiser that takes a snippet from an Australian legal document and synthesises a question related to the snippet, or a topic discussed therein, and an answer to that question, extracted from the snippet.

Your question must be decontextualised and standalone from the snippet. If the question pertains to a particular jurisdiction or document, it must state that explicitly (eg, 'In Victoria, is it lawful for ...?', 'What did the Court decide in Mabo v Queensland (No 2) [1992] HCA 23?', etc...).

Your answer must also be decontextualised and standalone from the snippet. It must reference the document from which it came (eg, 'Under the Crimes Act 1958 (Vic), ...', 'In Mabo v Queensland (No 2) [1992] HCA 23, the Court decided ...', etc...), not the snippet itself. It must be capable of being understood on its own and without reference to the snippet or its source document.

When referring to a document (eg, the Crimes Act) or a part thereof (eg, Paragraph 1), or to a person (eg, the Minister), organisation (eg, the Department) or concept (eg, the rule of law), you must refer to it by its full name (eg, the Crimes Act 1958 (Vic) instead of the Crimes Act, Paragraph 1 of ABC v XYZ instead of Paragraph 1, the Commonwealth Minister for Finance instead of the Minister).

If it is not possible to synthesise a question and answer from the snippet, you must respond with `<!no_qa!>`. Otherwise, your response must conform to the provided format.
```

The resulting prompts were then sent to `gpt-4` with the following hyperparameters:
| Hyperparameter | Value |
| --- | --- |
| `temperature` | 0 |
| `top_p` | 1 |
| `frequency_penalty` | 0 |
| `presence_penalty` | 0 |
| `max_tokens` | 768 |

`gpt-4`'s responses were parsed with the regex pattern `#\s?Question:?\s+((?:\n|.)+)#\s?Answer:?\s+((?:\n|.)+)`, yielding the question-answer pairs. Any malformed responses were discarded.

## Changelog 🔄
All notable changes to the dataset are documented in its [Changelog 🔄](https://huggingface.co/datasets/umarbutler/open-australian-legal-qa/blob/main/CHANGELOG.md).

This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Licence 📜
The dataset is distributed under the same licence as the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus/blob/main/LICENCE.md).

## Citation 🔖
If you've relied on the dataset for your work, please cite:
```latex
@misc{butler-2023-open-australian-legal-dataset,
    author = {Butler, Umar},
    year = {2023},
    title = {Open Australian Legal QA},
    publisher = {Hugging Face},
    version = {2.0.0},
    doi = {10.57967/hf/1479},
    url = {https://huggingface.co/datasets/umarbutler/open-australian-legal-qa}
}
```

## Acknowledgements 🙏
In the spirit of reconciliation, the author acknowledges the Traditional Custodians of Country throughout Australia and their connections to land, sea and community. He pays his respect to their Elders past and present and extends that respect to all Aboriginal and Torres Strait Islander peoples today.

The author thanks Matthew Altenberg, who gave him the idea of using `gpt-4` to synthesise questions and answers from the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus).

The author also acknowledges the creators of the many Python libraries relied upon in the creation of the dataset.

Finally, the author is eternally grateful for the endless support of his wife and her willingness to put up with many a late night spent writing code and quashing bugs.