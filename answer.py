#Importing libraries
import pandas as pd
from transformers.pipelines import pipeline

#Using Bert pertained model
hg_comp = pipeline('question-answering', model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer="bert-large-cased-whole-word-masking-finetuned-squad")

#Reading the csv into a dataframe
data = pd.read_csv('examples.csv')

#To iterate over each row
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print("The answer for question ", (idx+1) , " is ", answer)
