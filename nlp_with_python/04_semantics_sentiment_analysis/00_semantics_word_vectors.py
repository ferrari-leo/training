import spacy
from scipy import spatial

nlp = spacy.load("en_core_web_lg")
# word vector
print(nlp("fox").vector)
# sentence vector
print(nlp("The quick brown fox jumped").vector)

# cosine similarity
tokens = nlp("lion cat pet")
tokens = nlp("like love hate")

for t1 in tokens:
    for t2 in tokens:
        print(t1.text, t2.text, t1.similarity(t2))

# vocab
print(nlp.vocab.vectors.shape)

tokens = nlp("dog cat nargle")
for t in tokens:
    print(t.text, t.has_vector, t.vector_norm, t.is_oov)

# vector arithmetic

cosine_similarity = lambda v1, v2: 1 - spatial.distance.cosine(v1, v2)

king = nlp.vocab["king"].vector
man = nlp.vocab["man"].vector
woman = nlp.vocab["woman"].vector

new_vector = king - man + woman
computed_similarities = []
for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([t[0].text for t in computed_similarities[:10]])
