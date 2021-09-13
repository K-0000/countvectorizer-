import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

tag= pd.read_csv("user_movies.csv")
movie= pd.read_csv("movies.csv")

  
data = pd.merge(tag,movie)
print(data)
data = pd.DataFrame(data)
data = data[['userId','title','genres']]
print(data)
vectorizer = CountVectorizer()
tf = TfidfVectorizer()
count_g = vectorizer.fit_transform(data['genres'])

print("_________________________________________")
print ("----Cosine_similarity calculatation----")
print("_________________________________________")
cosine = cosine_similarity(count_g,count_g)
lmao = list(enumerate(cosine[21]))
lmao = sorted(lmao, key=lambda x: x[1], reverse=True)
lmao = lmao[1:20]
print(lmao)
lol= [i[0] for i in lmao]
print (lol)
kek= data["title"].iloc[lol]
print (kek)

print("_________________________________________")
print ("----Cosine_similarity calculatation----")
print("_________________________________________")
cosine = cosine_similarity(count_g,count_g)
lmao = list(enumerate(cosine[21]))
lmao = sorted(lmao, key=lambda x: x[1], reverse=True)
lmao = lmao[1:20]
print(lmao)
lol= [i[0] for i in lmao]
print (lol)
kek= movie["title"].iloc[lol]
print (kek)

print("_________________________________________")
print ("----linear_kernel calculatation----")
print("_________________________________________")
cosine = linear_kernel(count_g,count_g)
lmao = list(enumerate(cosine[21]))
lmao = sorted(lmao, key=lambda x: x[1], reverse=True)
lmao = lmao[1:20]
print(lmao)
lol= [i[0] for i in lmao]
print (lol)
kek= data["title"].iloc[lol]
print (kek)

print("_________________________________________")
print ("----linear_kernel calculatation----")
cosine = linear_kernel(count_g,count_g)
lmao = list(enumerate(cosine[21]))
lmao = sorted(lmao, key=lambda x: x[1], reverse=True)
lmao = lmao[1:20]
print("_________________________________________")
print(lmao)
lol= [i[0] for i in lmao]
print (lol)
kek= movie["title"].iloc[lol]
print (kek)
