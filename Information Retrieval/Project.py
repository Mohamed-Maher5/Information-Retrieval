# Import Libraries
import nltk
import os
from nltk.tokenize import word_tokenize
from natsort import natsorted  
from nltk.stem import PorterStemmer
import math
import pandas as pd

nltk.data.path.append("/path/to/nltk_data")
nltk.download('punkt')


# Identify Folder Name
document_folder = "Document_collection"


# Store Data 
data = []
for filename in natsorted(os.listdir(document_folder)):
    file_path = os.path.join(document_folder, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        document_content = file.read()
        data.append(document_content)

print("\nData:")
for doc in data:
    print(doc)
print("\n")


# Tokenization
def Tokenization(data):
    tokenized_documents = []
    for doc in data:
        tokens = word_tokenize(doc)
        tokenized_documents.append(tokens)
    return tokenized_documents

tokenized_documents = Tokenization(data)

print("Tokenized Docs:")
for doc_index,tokens in enumerate(tokenized_documents):
    print(f"Document {doc_index+1}: {tokens}")
print("\n")


# Stemming
def Stemming(tokenized_documents):
    stemmed_documents = []
    stemmer = PorterStemmer()
    for document_tokens in tokenized_documents:
        stemmed_tokens = [stemmer.stem(token) for token in document_tokens]
        stemmed_documents.append(stemmed_tokens)
    return stemmed_documents

stemmed_documents = Stemming(tokenized_documents)

print("Stemmed Docs:")
for document_index, document_tokens in enumerate(stemmed_documents):
    print(f"Document {document_index+1}: {document_tokens}")
print("\n")


# Constructing Auxiliary structure(s) (Positional index)
def create_positional_index(stemmed_documents):
    positional_index = {}
    for doc_index, doc_tokens in enumerate(stemmed_documents, start=1):
        for position, token in enumerate(doc_tokens, start=1):
            if token not in positional_index:
                positional_index[token] = [0, {}]
            if doc_index not in positional_index[token][1]:
                positional_index[token][1][doc_index] = [position]
                positional_index[token][0] += 1
            else:
                positional_index[token][1][doc_index].append(position)
    return positional_index

positional_index = create_positional_index(stemmed_documents)

print("Positional index:")
for key, value in positional_index.items():
    print(f"term: {key}, frequency: {value[0]}, posing_list: {value[1]}")
print("\n")


# PreProcessing function
def preprocess_phrase_query(phrase_query):
    phrase_query = Tokenization([phrase_query])
    phrase_query = Stemming(phrase_query)
    return phrase_query


# Phrase Query
def get_related_documents(phrase_query):
    documents = [[] for _ in range(10)]
    related_documents = []
    for word in phrase_query[0]:
        if word in positional_index.keys():
            for key in positional_index[word][1].keys():
                if documents[key - 1] != []:
                    if documents[key - 1][-1] == positional_index[word][1][key][0] - 1:
                        documents[key - 1].append(positional_index[word][1][key][0])
                else:
                    documents[key - 1].append(positional_index[word][1][key][0])

    for position, positions_list in enumerate(documents, start=1):
        if len(positions_list) == len(phrase_query[0]):
            related_documents.append([position,positions_list])
    return related_documents

phrase_query = input("Enter a Phrase Query:")
phrase_query = preprocess_phrase_query(phrase_query) 
phrase_query_related_documents = get_related_documents(phrase_query)

print("Related Docs:")
for doc, positions_lis in phrase_query_related_documents:
    print(f"Document {doc}: in positions: {positions_lis}")
print("\n")


# Term frequency (Raw)
terms = list(set([word for doc in stemmed_documents for word in doc]))
term_frequency = pd.DataFrame(index=terms, columns=['doc' + str(i) for i in range(1, 11)])
term_frequency = term_frequency.fillna(0)

for i, doc in enumerate(stemmed_documents, start=1):
    for word in doc:
        term_frequency.at[word, 'doc' + str(i)] += 1

print("Term Frequency:")
print(term_frequency)
print("\n")


# Term frequency (Weight)
Weighted_term_frequency = term_frequency.applymap(lambda x: math.log10(x) + 1 if x > 0 else 0)
print("Term frequency (Weight):")
print(Weighted_term_frequency)
print("\n")


# IDF
total_documents = 10  
idf = []
for term, document_positions in positional_index.items():
    idf.append([document_positions[0],math.log10(total_documents / document_positions[0])])

idf = pd.DataFrame(idf, columns=['df', 'idf'], index=positional_index.keys())
print("IDF:")
print(idf)
print("\n")


# TF-IDF
tf_idf = pd.DataFrame(index=terms, columns=['doc' + str(i) for i in range(1, 11)])
for key in idf.index:
   tf_idf.loc[key] =  Weighted_term_frequency.loc[key].multiply(idf.loc[key]['idf'])
print(("TF-IDF:"))
print(tf_idf)
print("\n")


# Document Length
doc_length = tf_idf.applymap(lambda x: x ** 2).sum()**0.5
print("Document Length:")
print(doc_length)
print('\n')


# Weighted tf_idf
weighted_tf_idf = pd.DataFrame(index=terms, columns=['doc' + str(i) for i in range(1, 11)])
doc_index = 1
for key in tf_idf.columns:
   weighted_tf_idf[key] =  tf_idf[key].div(doc_length['doc'+str(doc_index)])
   doc_index+=1
print("Weighted tf_idf:")
print(weighted_tf_idf)
print("\n")


# Get Similar Documents 
query = input("Enter a Query: ")
query = preprocess_phrase_query(query)
related_documents = get_related_documents(query)

similarity = []
for doc in related_documents:
    query_df = pd.DataFrame(index=query[0], columns=['tf-raw','tf-weighted','idf','tf-idf','weighted_tf-idf','doc_product'])
    doc_data = []
    for word in query[0]:
        word_data = []
        word_data.append(term_frequency.loc[word]['doc'+str(doc[0])])
        word_data.append(Weighted_term_frequency.loc[word]['doc'+str(doc[0])])   
        word_data.append(idf.loc[word][1])
        word_data.append(tf_idf.loc[word]['doc'+str(doc[0])])  
        doc_data.append(word_data)

    query_length = 0
    for word in range(len(query[0])):
        query_length += (doc_data[word][-1]**2)

    query_length = query_length**0.5
    word_index = 0
    for word in query[0]:
        doc_data[word_index].append(doc_data[word_index][-1]/query_length)
        doc_data[word_index].append(doc_data[word_index][-1]*weighted_tf_idf.loc[word]['doc'+str(doc[0])])
        query_df.iloc[word_index] = doc_data[word_index]
        word_index+=1

    similarity.append([doc[0],query_df['doc_product'].sum()])
    print('Document:',doc[0],'\n',query_df,'\nQuery Length',query_length,'\tSimilarity:',query_df['doc_product'].sum(),"\n\n")

sorted_similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
print("returned_docs: ", end='')
for doc in sorted_similarity:
    print('document',doc[0],' , ', end='')