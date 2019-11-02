import json
import pandas as pd
from wordcloud_fa import WordCloudFa
import matplotlib.pyplot as plt


stopwords = open("stopwords-ur.txt").read().splitlines()
stopwords=[i.lower() for i in stopwords]

dataset_path = "./data/dev-v2.0-generated.json"

def wordcloud(counter):
    """A small wordloud wrapper"""
    wc = WordCloudFa(width=1200, height=800,
                   background_color="white",
                   max_words=200)
    wc.generate_from_frequencies(counter)
    fig=plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("./Diagrams/most_common_words.jpg")
    plt.show()


context=[]
questions=[]
questio_with_no_answer=[]
no_of_paragraphs=[]
tokens=[]


with open(dataset_path) as f:
    dataset = json.load(f)['data']
f1 = exact_match = total = exact_sentence = inclusion = random = 0

for article in dataset:
    no_of_paragraphs.append(len(article['paragraphs']))
    for paragraph in article['paragraphs']:
        context.append(len(paragraph['context'].split()))
        tokens.append(paragraph['context'].split(' '))
        # print(len([x for x in paragraph['context'].split() if x in stopwords]))


        questions.append(len(paragraph['qas']))
        count = 0
        for qa in paragraph['qas']:
            if len(qa['answers'])==0:
                count+=1
        questio_with_no_answer.append(count)

tokens = [item for sublist in tokens for item in sublist]
tokens=[i for i in tokens if i.lower() not in stopwords]

from collections import Counter
counter = Counter(tokens)
#print(counter.most_common())

wordcloud(counter)
plt.bar([i for i in range(len(no_of_paragraphs))],no_of_paragraphs)
plt.xticks(rotation=70)
plt.title("Paragraph in each Title")
plt.xlabel("Title No.")
plt.ylabel("No.of paragraphs")
plt.savefig("./Diagrams/Paragraph_in_each_Title.jpg")
plt.show()

df=pd.DataFrame({'context_length':context,'total_questions':questions,"empty_answers":questio_with_no_answer})


plt.bar([i for i in range(len(df['context_length']))],df['context_length'])
plt.xticks(rotation=70)
plt.title("Context length")
plt.xlabel("Context")
plt.ylabel("Context Length")
plt.savefig("./Diagrams/Context_Length.jpg")
plt.show()



plt.bar([i for i in range(len(df["total_questions"]))],df["total_questions"])
plt.xlabel("Context")
plt.ylabel("NO. of Questions")
plt.title("Number of Question in each context")
plt.savefig("./Diagrams/Number_of_Question_in_each_context.jpg")
plt.show()


plt.bar([i for i in range(len(df['empty_answers']))],df['empty_answers'])
plt.xlabel("Context")
plt.ylabel("NO. of Question without answer")
plt.title("Number of Questions without Answer in each context")
plt.savefig("./Diagrams/Number_of_Questions_without_Answer.jpg")
plt.show()
