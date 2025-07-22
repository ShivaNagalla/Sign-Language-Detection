from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import xml.etree.ElementTree as ET
import re

def clean_phrase(text):
    text = text.lower()
    text = re.sub(r'\[silence\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_corpus_file(corpus_path):
    print(f'Parsing corpus file: {corpus_path}')
    tree = ET.parse(corpus_path)
    root = tree.getroot()
    
    sequences = []
    recording_ids = []
    
    for recording in root.findall('recording'):
        rec_id = recording.attrib.get('name')
        segment = recording.find('segment')
        if segment is not None:
            orth = segment.find('orth')
            if orth is not None:
                phrase = orth.text
                cleaned_phrase = clean_phrase(phrase)
                sequences.append(cleaned_phrase)
                recording_ids.append(rec_id)
    return recording_ids, sequences

def build_and_save_tokenizer(phrases, tokenizer_path='tokenizer.json', oov_token='<OOV>', num_words=None):
    """
    Build a tokenizer on the list of phrases and save it to a JSON file.
    Args:
        phrases (list[str]): List of cleaned phrase strings.
        tokenizer_path (str): Path to save the tokenizer JSON.
        oov_token (str): Token for out-of-vocabulary words.
        num_words (int or None): Limit vocabulary size. None = no limit.
    Returns:
        tokenizer: The fitted Keras Tokenizer instance.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters='')
    tokenizer.fit_on_texts(phrases)
    
    # Save tokenizer to JSON
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")  # +1 for padding token 0
    return tokenizer

def phrases_to_padded_sequences(tokenizer, phrases, max_len=None):
    """
    Convert phrases to sequences of token IDs and pad them.
    Args:
        tokenizer: A fitted Keras Tokenizer.
        phrases (list[str]): List of cleaned phrase strings.
        max_len (int or None): Max length to pad/truncate sequences. If None, use max phrase length.
    Returns:
        padded_sequences: numpy array of shape (num_samples, max_len)
    """
    sequences = tokenizer.texts_to_sequences(phrases)
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    print(f"Padded sequences shape: {padded_sequences.shape}")
    return padded_sequences

def main():
    corpus_path = r'C:\Users\manug\Sign-Language-Detection\rwth-boston-104\corpus\train.sentences.pronunciations.corpus'
    recording_ids, sequences = parse_corpus_file(corpus_path)
    print(f'Parsed {len(sequences)} phrases')

    tokenizer = build_and_save_tokenizer(sequences)
    padded_sequences = phrases_to_padded_sequences(tokenizer, sequences)
    print('Example tokenized phrase:', padded_sequences[0])

if __name__ == '__main__':
    main()