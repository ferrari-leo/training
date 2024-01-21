import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

npr = pd.read_csv("npr.csv")

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")

dtm = tfidf.fit_transform(npr["Article"])

nmf_model = NMF(n_components=7, random_state=42)

nmf_model.fit(dtm)

for index, topic in enumerate(nmf_model.components_):
    print(f"The top 15 words for topic #{index}")
    print([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print("\n\n")

topic_results = nmf_model.transform(dtm)

npr["Topic"] = topic_results.argmax(axis=1)
