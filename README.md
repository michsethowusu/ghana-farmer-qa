---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: category
    dtype: string
  - name: source_text
    dtype: string
  - name: original_twi
    dtype: string
  - name: file_name
    dtype: string
  source_datasets:
  - ghananlpcommunity/twi-english-agric
language:
- en
- tw
tags:
- agriculture
- qa
- ghana
- farmer-qa
- synthetic-data
- gemini
---

# Ghana Farmer Q&A Dataset

This dataset contains over 151,000 high-quality question-and-answer (Q&A) pairs generated for farmers in Ghana, derived from the `ghananlpcommunity/twi-english-agric` dataset.

## Dataset Structure

Each row represents an independent Q&A pair grounded in agricultural context:
- `question`: A first-person question as if a farmer is asking an extension officer.
- `answer`: Detailed knowledge and explanation answering the question.
- `category`: The thematic category of the agricultural topic (e.g., Basic Field Setup, Plant Health, Harvest & Money).
- `source_text`: The English translation of the source audio transcription (`translated_english`).
- `original_twi`: The original Twi transcription.
- `file_name`: Source audio file identifier.

## Generation Methodology
Generated using **Gemini-3.6-flash** with strict prompt instructions ensuring questions stand alone without requiring prior context.
