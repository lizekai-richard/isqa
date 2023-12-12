import re
import string
import nltk
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

print("Download Done!")


def replace_edited_word(df):
    edited_df = df
    for i in range(len(df)):
        original_text = df.iloc[i, 1]
        edit_word = df.iloc[i, 2]
        edited_text = re.sub(r'<\S*\s*\S*>', edit_word, original_text)
        edited_df.iloc[i, 1] = edited_text

    return edited_df


# lowercase
def lowercase(text):
    return text.lower()


# remove numbers:
def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# tokenization
def tokenization(text):
    tokenized_text = word_tokenize(text)
    return tokenized_text


# remove punctuations
def remove_punctuation(tokenized_text):
    edited_tokens = []
    for token in tokenized_text:
        if token not in string.punctuation:
            edited_tokens.append(token)
    return edited_tokens


# remove stopwords
def remove_stopwords(tokenized_text):
    edited_tokens = []
    stopword_list = stopwords.words('english')
    for token in tokenized_text:
        if token not in stopword_list:
            edited_tokens.append(token)
    return edited_tokens


# True-casing
def true_casing(tokenized_text):
    edited_tokens = []
    pos_tokens = pos_tag(tokenized_text)
    for pos_token in pos_tokens:
        token = pos_token[0].lower()
        tag = pos_tokens[1]
        if tag == 'NNP':
            edited_tokens.append(token.capitalize())
        else:
            edited_tokens.append(token)
    return edited_tokens

# Stemming
def stemming(tokenized_text):
    edited_tokens = []
    stemmer = PorterStemmer()
    for token in tokenized_text:
        edited_tokens.append(stemmer.stem(token))
    
    return edited_tokens


# Lemmanisation
def lemmanisation(tokenized_text):
    edited_tokens = []
    lemmaniser = WordNetLemmatizer()
    for token in tokenized_text:
        edited_tokens.append(lemmaniser.lemmatize(token))

    return edited_tokens


def remove_special_tokens(text):
    
    return re.sub(r'<pad>|<\\s>', '', text)


def process_example(text):
    return lemmanisation(
        remove_stopwords(
            remove_stopwords(
                remove_punctuation(
                    tokenization(
                        remove_special_tokens(
                            lowercase(text)
                        )
                    )
                )
            )
        )
    )


def process(df):
    df['original'] = df['original'].apply(lowercase)
    df['original'] = df['original'].apply(remove_numbers)
    df['original'] = df['original'].apply(tokenization)
    df['original'] = df['original'].apply(remove_punctuation)
    df['original'] = df['original'].apply(remove_stopwords)
    # df['original'] = df['original'].apply(stemming)
    df['original'] = df['original'].apply(lemmanisation)
    return df


if __name__ == '__main__':

    # load data
    train_df = pd.read_csv('./data/subtask-1/train.csv')
    dev_df = pd.read_csv('./data/subtask-1/dev.csv')
    test_df = pd.read_csv('./data/subtask-1/test.csv')

    train_df = replace_edited_word(train_df)
    dev_df = replace_edited_word(dev_df)
    test_df = replace_edited_word(test_df)

    train_df = process(train_df)
    dev_df = process(dev_df)
    test_df = process(test_df)

    train_tokenized_corpus = train_df['original']
    dev_tokenized_corpus = dev_df['original']
    test_tokenized_corpus = test_df['original']

    print(train_tokenized_corpus)

    


