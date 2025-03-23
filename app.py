import streamlit as st
import nltk
import re
import random
import time
import requests
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from keybert import KeyBERT
import spacy
from spacy.cli import download

# Download necessary NLTK and spacy resources
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt')
# Uncomment the next line if you haven't downloaded the spacy model yet:
# download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

# Load models and tokenizers
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
t5_model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
keybert_model = KeyBERT(model="distilbert-base-nli-mean-tokens")

# ----------------------------
# Helper Functions
# ----------------------------

def summarize(text, maxSummarylength=2000):
    inputs = bart_tokenizer.encode("summarize: " + text,
                                   return_tensors="pt",
                                   max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs,
        max_length=int(maxSummarylength),
        min_length=int(maxSummarylength / 5),
        length_penalty=10.0,
        num_beams=4,
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def split_text_into_pieces(text, max_tokens=3000, overlapPercent=10):
    tokens = bart_tokenizer.tokenize(text)
    overlap_tokens = int(max_tokens * overlapPercent / 100)
    pieces = [tokens[i:i + max_tokens]
              for i in range(0, len(tokens),
                             max_tokens - overlap_tokens)]
    text_pieces = [bart_tokenizer.decode(
        bart_tokenizer.convert_tokens_to_ids(piece),
        skip_special_tokens=True) for piece in pieces]
    return text_pieces

def recursive_summarize(text, max_length=2000):
    # Note: Debug prints have been removed for clarity in the Streamlit UI.
    tokens = bart_tokenizer.tokenize(text)
    expected_chunks = max(1, len(tokens) / max_length)
    new_max_length = int(len(tokens) / expected_chunks) + 2

    pieces = split_text_into_pieces(text, max_tokens=new_max_length)
    summaries = []
    for piece in pieces:
        summary_piece = summarize(piece, maxSummarylength=int(new_max_length / 3 * 2))
        summaries.append(summary_piece)
    concatenated_summary = ' '.join(summaries)
    tokens_summary = bart_tokenizer.tokenize(concatenated_summary)

    if len(tokens_summary) > max_length:
        return recursive_summarize(concatenated_summary, max_length=max_length)
    else:
        if len(pieces) > 1:
            final_summary = summarize(concatenated_summary, maxSummarylength=max_length)
        else:
            final_summary = concatenated_summary
        return final_summary

def keyword_extract(article):
    start_time = time.time()
    keywords = keybert_model.extract_keywords(
        article,
        top_n=200,
        keyphrase_ngram_range=(1, 1),
        stop_words="english"
    )
    st.write(f"Keyword extraction executed in {time.time() - start_time:.4f} seconds.")
    return keywords

def corresponding_sentences(text, filtered_keywords):
    sentences = sent_tokenize(text)
    corr_sentences = dict()
    for keyword in filtered_keywords:
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                corr_sentences[keyword] = sentence
                break
    return corr_sentences

# POS mapping for wordnet
POS_MAP = {
    'VERB': wn.VERB,
    'NOUN': wn.NOUN,
    'PROPN': wn.NOUN
}

def get_wordsense(sent, word):
    word_lower = word.lower().replace(" ", "_")
    synsets = wn.synsets(word_lower, 'n')
    if synsets:
        try:
            # Use a simple Lesk algorithm as a fallback
            sense = nltk.wsd.lesk(sent.split(), word, 'n')
            return sense
        except Exception:
            return None
    else:
        return None

def get_distractors_wordnet(syn, word):
    distractors = []
    word_lower = word.lower().replace(" ", "_")
    hypernyms = syn.hypernyms()
    if not hypernyms:
        return distractors
    for item in hypernyms[0].hyponyms():
        name = item.lemmas()[0].name()
        if name.lower() == word_lower:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name and name not in distractors:
            distractors.append(name)
    return distractors

def get_distractors_conceptnet(word):
    word_lower = word.lower().replace(" ", "_")
    distractor_list = []
    url = f"http://api.conceptnet.io/query?node=/c/en/{word_lower}/n&rel=/r/PartOf&start=/c/en/{word_lower}&limit=5"
    obj = requests.get(url).json()
    for edge in obj.get('edges', []):
        link = edge['end']['term']
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2.get('edges', []):
            word2 = edge['start']['label']
            if word_lower not in word2.lower() and word2 not in distractor_list:
                distractor_list.append(word2)
    return distractor_list

def generate_mcq(keyword_sentences, key_distractor_list):
    output_lines = []
    index = 1
    for keyword, sentence in keyword_sentences.items():
        if keyword not in key_distractor_list:
            output_lines.append(f"Skipping '{keyword}': No distractors found.")
            continue

        # Create a regex pattern to find the keyword (case-insensitive)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        distractors = key_distractor_list[keyword]
        choices = [keyword.capitalize()] + distractors

        # Replace the keyword with a fixed blank "_______"
        masked_sentence = pattern.sub("_______", sentence)
        output_lines.append(f"{index}) {masked_sentence}")

        top4choices = choices[:4]
        random.shuffle(top4choices)
        optionchoices = ['a', 'b', 'c', 'd']
        for idx, choice in enumerate(top4choices):
            output_lines.append(f"\t{optionchoices[idx]}) {choice}")
        output_lines.append("\nMore options: " + str(choices[4:]) + "\n")
        index += 1

    return "\n".join(output_lines)


# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("MCQ Generator from Text")
st.write("Upload a text file or paste your text below, then click the button to generate MCQs.")

# Option to upload a file or enter text manually
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file is not None:
    article = uploaded_file.read().decode("utf-8")
else:
    article = st.text_area("Enter the text here", height=200)

if st.button("Generate MCQs"):
    if not article.strip():
        st.error("Please provide some text for processing.")
    else:
        with st.spinner("Processing text..."):
            # Step 1: Summarize text recursively
            summary = recursive_summarize(article)
            st.subheader("Summary")
            st.write(summary)

            # Step 2: Extract keywords
            original_keywords = keyword_extract(article)
            required_keywords = [i[0] for i in original_keywords]

            # Filter keywords that appear in the summary
            filtered_keywords = {kw for kw in required_keywords if kw.lower() in summary.lower()}

            # Get corresponding sentences for filtered keywords
            keyword_sentences = corresponding_sentences(summary, filtered_keywords)

            # Generate distractors for each keyword
            key_distractor_list = {}
            for keyword, sentence in keyword_sentences.items():
                sense = get_wordsense(sentence, keyword)
                if sense:
                    distractors = get_distractors_wordnet(sense, keyword)
                    if not distractors:
                        distractors = get_distractors_conceptnet(keyword)
                    if distractors:
                        key_distractor_list[keyword] = distractors
                else:
                    distractors = get_distractors_conceptnet(keyword)
                    if distractors:
                        key_distractor_list[keyword] = distractors

            # Generate MCQs based on the sentences and distractors
            mcq_output = generate_mcq(keyword_sentences, key_distractor_list)

        st.subheader("Generated MCQs")
        st.text(mcq_output)
