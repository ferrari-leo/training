import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# region Part 1
npr = pd.read_csv("npr.csv")
display(npr.head())

cv = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
dtm = cv.fit_transform(npr["Article"])

LDA = LatentDirichletAllocation(n_components=7, random_state=42)
LDA.fit(dtm)
# endregion
# region Part 2
# grab words
cv.get_feature_names_out()

# grab topics
print(LDA.components_.shape)
single_topic = LDA.components_[0]
top_ten_words = single_topic.argsort()[-10:]
for index in top_ten_words:
    print(cv.get_feature_names_out()[index])

# grab highest prob words per topic
for index, topic in enumerate(LDA.components_):
    print(f"The top 15 words for topic #{index}")
    print([cv.get_feature_names_out()[index] for index in topic.argsort()[-15:]])
    print("\n\n")

# assign topic number to article
# produce matrix where each row is the probability distribution of
# a document belonging to a topic
topic_results = LDA.transform(dtm)
npr["Topic"] = topic_results.argmax(axis=1)

# endregion
