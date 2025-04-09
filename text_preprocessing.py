import re
import string
from bs4 import BeautifulSoup
import spacy

def lower_number_punctuation(text):
    result = text.lower()
    result = re.sub(r'\d+', '', result)
    return result.translate(str.maketrans('', '', string.punctuation))

def remove_whitespace(text):
    return " ".join(text.split())

def remove_html(text):
    return BeautifulSoup(text, features='html.parser').get_text()

def remove_urls(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_stopwords(text):
    nlp = spacy.load("en_core_web_sm")
    if isinstance(text,str):
        chunks = []
        for i in range(0, len(text), 20000):
            chunk = text[i:i+ 20000]
            original = nlp(chunk)
            filtered = [token.text for token in original if not token.is_stop]
            chunks.append(' '.join(filtered))
        return ' '.join(chunks)
    return ' '

def remove_html_tags(text):
  if isinstance(text, (str, bytes)):
      clean = re.compile('<.*?>') 
      return re.sub(clean, '', text)
  return text

def remove_urls(text):
  if isinstance(text, (str, bytes)):
    url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
    return url_pattern.sub('', text) 
  return text

def process_text(text):
    input_text = text
    try:
        input_text = remove_html(input_text)
        input_text = remove_urls(input_text)
        input_text = lower_number_punctuation(input_text)
        input_text = remove_whitespace(input_text)
        input_text = remove_stopwords(input_text)
    except Exception as error:
        print(error)
    finally:
        print(input_text)
    
    return input_text