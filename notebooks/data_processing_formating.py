import re
import pandas as pd
import os
from tqdm import tqdm

def clean_financial_text(text):
    """
    Clean financial text according to the plan specifications:
    - Remove boilerplate legal language
    - Normalize company ticker mentions
    - Remove unwanted characters
    - Filter documents by length
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove special characters but keep financial symbols like $, %, ., ,, -, (, )
    # Allow alphanumeric, whitespace, and the specified financial symbols
    text = re.sub(r'[^\w\s\$\%\.\,\-\(\)]', ' ', text)


    # Normalize ticker mentions (uppercase known tickers) - Expanded list
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'XOM',
               'NVDA', 'META', 'BRK.B', 'UNH', 'LLY', 'AVGO', 'JPM', 'XOM', 'V', 'UNH',
               'MA', 'PG', 'HD', 'MRK', 'ORCL', 'COST', 'ABBV', 'CVX', 'PEP', 'BAC']
    for ticker in tickers:
        # Fix the f-string syntax by constructing the pattern string differently
        # Escape the dot outside the f-string expression
        pattern = r'\b' + re.escape(ticker.lower()) + r'\b'
        text = re.sub(pattern, ticker.replace('.', ''), text, flags=re.IGNORECASE)


    # Remove documents that are too short or too long (as per plan)
    if len(text) < 10 or len(text) > 10000: # Adjusted max length for potentially shorter QA pairs
        return None

    return text

def extract_financial_sections(text):
    """
    Extract key financial information sections (simplified for core text cleaning subtask)
    """
    sections = {}
    sections['text_length'] = len(text) if text else 0
    sections['word_count'] = len(text.split()) if text else 0

    # Basic revenue mention extraction as before
    revenue_patterns = [
        r'revenue.*?(\$[\d,]+\.?\d*\s*(?:million|billion|thousand)?)',
        r'net\s+sales.*?(\$[\d,]+\.?\d*\s*(?:million|billion|thousand)?)',
        r'total\s+revenue.*?(\$[\d,]+\.?\d*\s*(?:million|billion|thousand)?)'
    ]
    revenue_mentions = []
    if text:
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            revenue_mentions.extend(matches)
    sections['revenue_mentions'] = revenue_mentions

    return sections

def process_financial_datasets():
    processed_data = []
    project_path = "/content/drive/MyDrive/financial-llm-finetuning" # Define project_path here

    # Process FinanceQA dataset (numerical reasoning QA)
    financeqa_path = os.path.join(project_path, 'data/raw/financeqa.csv')
    if os.path.exists(financeqa_path):
        try:
            df_fin_qa = pd.read_csv(financeqa_path)
            print(f"📊 Processing FinanceQA dataset: {len(df_fin_qa)} rows")

            for idx, row in tqdm(df_fin_qa.iterrows(), desc="Cleaning FinanceQA"):
                try:
                    # Extract relevant columns
                    question = str(row.get('Open-ended Verifiable Question', ''))
                    cot = str(row.get('Complex_CoT', ''))
                    response = str(row.get('Response', '')) # Using Response for the answer

                    # Clean text
                    cleaned_question = clean_financial_text(question)
                    cleaned_cot = clean_financial_text(cot)
                    cleaned_response = clean_financial_text(response)


                    if cleaned_question and cleaned_response: # Ensure question and response are valid
                         # Combine question, context, and response for instruction tuning format
                        processed_entry = {
                            'source': 'financeqa',
                            'instruction': cleaned_question,
                            'input': cleaned_cot if cleaned_cot else "", # Use CoT as input, or empty string if not available/cleaned
                            'output': cleaned_response,
                            'metadata': extract_financial_sections(cleaned_question + " " + (cleaned_cot if cleaned_cot else "") + " " + cleaned_response)
                        }
                        processed_data.append(processed_entry)
                    # else:
                        # print(f"Skipping FinanceQA row {idx} due to cleaning producing empty text.")

                except Exception as e:
                    print(f"⚠️ Error processing FinanceQA row {idx}: {e}")

        except Exception as e:
            print(f"⚠️ Error loading or processing FinanceQA dataset: {e}")


    # Process Financial PhraseBank dataset (sentiment analysis)
    financial_phrasebank_path = os.path.join(project_path, "data/raw/financial_phrasebank.csv")
    if os.path.exists(financial_phrasebank_path):
        try:
            df_fin_ph = pd.read_csv(financial_phrasebank_path)
            print(f"\n📊 Processing Financial PhraseBank dataset: {len(df_fin_ph)} rows")

            for idx, row in tqdm(df_fin_ph.iterrows(), desc="Cleaning Financial PhraseBank"):
                try:
                    # Extract relevant columns
                    sentence = str(row.get('sentence', ''))
                    sentiment = str(row.get('sentiment_text', ''))

                    # Clean text
                    cleaned_sentence = clean_financial_text(sentence)

                    if cleaned_sentence and sentiment: # Ensure sentiment is also present
                        processed_entry = {
                            'source': 'financial_phrasebank',
                            'instruction': "Analyze the sentiment of this sentence:", # Example instruction
                            'input': cleaned_sentence,
                            'output': sentiment, # The sentiment label is the output
                            'metadata': extract_financial_sections(cleaned_sentence)
                        }
                        processed_data.append(processed_entry)
                    # else:
                        # print(f"Skipping Financial PhraseBank row {idx} due to cleaning producing empty text or missing sentiment.")

                except Exception as e:
                     print(f"⚠️ Error processing Financial PhraseBank row {idx}: {e}")

        except Exception as e:
             print(f"⚠️ Error loading or processing Financial PhraseBank dataset: {e}")


    # Process FinTextQA dataset (contextual QA)
    fintextqa_path = os.path.join(project_path, "data/raw/fintextqa.csv")
    if os.path.exists(fintextqa_path):
        try:
            df_txt_qa = pd.read_csv(fintextqa_path)
            print(f"\n📊 Processing FinTextQA dataset: {len(df_txt_qa)} rows")

            for idx, row in tqdm(df_txt_qa.iterrows(), desc="Cleaning FinTextQA"):
                try:
                    # Extract relevant columns
                    question = str(row.get('question', ''))
                    answer = str(row.get('answer', ''))
                    context = str(row.get('context', ''))

                    # Clean text
                    cleaned_question = clean_financial_text(question)
                    cleaned_answer = clean_financial_text(answer)
                    cleaned_context = clean_financial_text(context)

                    # For contextual QA, input is context + question, output is answer
                    if cleaned_question and cleaned_answer and cleaned_context:
                        processed_entry = {
                            'source': 'fintextqa',
                            'instruction': cleaned_question,
                            'input': cleaned_context,
                            'output': cleaned_answer,
                            'metadata': extract_financial_sections(cleaned_context + " " + cleaned_question + " " + cleaned_answer)
                        }
                        processed_data.append(processed_entry)
                    # else:
                        # print(f"Skipping FinTextQA row {idx} due to cleaning producing empty text in question, answer, or context.")

                except Exception as e:
                    print(f"⚠️ Error processing FinTextQA row {idx}: {e}")

        except Exception as e:
             print(f"⚠️ Error loading or processing FinTextQA dataset: {e}")


    return processed_data

# Execute the processing
processed_data = process_financial_datasets()
print(f"\n✅ Finished processing datasets. Total processed entries: {len(processed_data)}")