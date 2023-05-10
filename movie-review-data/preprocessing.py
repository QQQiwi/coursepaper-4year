import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('omw-1.4')


def remove_pattern(input_txt, pattern):
    """Удаление ненужных частей текстов из элементов выборки"""
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt


def strip_html(text):
    """Преобразовать обрабатываемый текст в формат html библиотеки обработки
    текста beautifulSoup"""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def to_unicode(text):
    """Преобразование текста в кодировку UNICODE с учетом целых цифр и цифр с
    плавающей точкой"""
    if isinstance(text, float) or isinstance(text, int):
        text = str(text)
    if not isinstance(text, str):
        text = text.decode('utf-8', 'ignore')
    return text


def deleteEmoji(text):
    """Удаление эмодзи"""
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def remove_specchars_brackets_spaces(text):
    """Удаление спецсимволов, скобок и пробелов"""
    text = re.sub('\[[^]]*\]', ' ', text)
    text = re.sub(r'[^a-zA-z0-9\s]',' ',text)
    while "  " in text:
        text = re.sub('  ', ' ', text)
    return text


def delete_noise_text(text):
    """Удаление различных 'шумов' из текста"""
    text = to_unicode(text)
    soup = BeautifulSoup(text, "html.parser")
    text = strip_html(text)
    text = re.sub(r"http\S+", " ", text)
    text = deleteEmoji(text)
    text = text.encode('ascii', 'ignore')
    text = to_unicode(text)
    text = remove_specchars_brackets_spaces(text)
    text = text.lower() # замена больших букв на маленькие
    return text

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    """Удаление стоп-слов английского языка из текста"""
    # stop = set(stopwords.words('english'))

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def simple_stemmer(text):
    """Применение стемминга"""
    ps = SnowballStemmer(language='english')
    return ' '.join([ps.stem(word) for word in tokenizer.tokenize(text)])


def lemmatize_all(sentence):
    """Генератор для лемматизации текста"""
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        else:
            yield word


def lemmatize_text(text):
    """Лемматизация текста"""
    return ' '.join(lemmatize_all(text))


def make_wordcloud(df):
    """Создание облака слов из текстов выборки, построение гистограммы по
    классам целевого признака"""
    all_words = " ".join([sentence for sentence in df['review']])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png', dpi=300)



def main():
    csv_filepath = "review-data/IMDB-Dataset.csv"
    df = pd.read_csv(csv_filepath)
    df['review'] = np.vectorize(remove_pattern)(df['review'], "@[\w]*")
    df['review'] = df['review'].apply(delete_noise_text)
    df['review'] = df['review'].apply(remove_stopwords)
    df['review'] = df['review'].apply(simple_stemmer)
    df['review'] = df['review'].apply(lemmatize_text)
    df.to_csv("processed_review.csv", index=False)

    df = pd.read_csv("processed_review.csv")
    make_wordcloud(df)


if __name__ == "__main__":
    main()
