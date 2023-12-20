#region Spacy basics
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

for token in doc: print(token.text, token.pos, token.pos_, token.dep_)

print(nlp.pipeline)
print(nlp.pipe_names)

doc2 = nlp(u"Tesla isn't looking into startups anymore")
for token in doc2: print(token.text, token.pos, token.pos_, token.dep_)

doc3 = nlp(u'Although commonly attributed to John Lennon from his song "Beautiful Boy", \
           the phrase "Life is what happens to us while we are making other plans" was written by \
           cartoonist Allen Saunders and published in Reader\'s digest in 1957, when Lennon was 17.')

life_quote = doc3[17:31]
print(life_quote)
print(type(life_quote)) # this is a span of a larger doc
print(type(doc3))

doc4 = nlp(u'This is the first sentence. This is anothe sentence. This is the last sentence.')
# can separate sentences and identify sentence starters
for sentence in doc4.sents: print(sentence)
print(doc4[6].is_sent_start)
print(doc4[7].is_sent_start)
#endregion