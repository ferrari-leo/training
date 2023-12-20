import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

#region Part 1
mystring = '"We\'re moving to L.A.!"'
print(mystring)

doc = nlp(mystring)
for token in doc: print(token.text)

doc2 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us as http://www.oursite.com!")
for t in doc2: print(t)

doc3 = nlp(u"A 5km NYC cab ride costs $10.30")
for t in doc3: print(t)

doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")
for t in doc4: print(t)
print(len(doc4))
# vocab objects:
print(len(doc4.vocab))

doc5 = nlp(u"It is better to give than receive.")
print(doc5[0])
print(doc5[2:5])
# note:spacy doesn't support item reassignment
doc5[0] = 'test'

# finds named entities
doc8 = nlp(u"Apple to build a Hong Kong factory for $6 million")
for token in doc8: print(token.text, end = ' | ')
for entity in doc8.ents:
    print(entity) 
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print('\n')

# noun chunks
doc9 = nlp(u"Autonomous cars shift insurance liability towards manufacturers.")
for chunk in doc9.noun_chunks: print(chunk)
#endregion
#region Part 2
# visualise - more info at https://spacy.io/usage/visualizers
doc = nlp(u"Apple is going to build a U.K. factory for $6 million.")
displacy.render(
    doc,
    style = 'dep', # syntactic dependency
    jupyter = True, 
    options = {'distance' : 110}
    )

doc = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.")
displacy.render(
    doc, 
    style = 'ent', # call out entities
    jupyter = True
)

# for rendering outside of jupyter - print result said 'serving on port 5000',
# so in browser visit http://127.0.0.1:5000/
doc = nlp(u"This is a sentence")
displacy.serve(
    doc,
    style = 'dep'
)
#endregion