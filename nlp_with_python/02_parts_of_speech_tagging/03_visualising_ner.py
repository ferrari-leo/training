import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million."
          u"By contrast, Sony only sold 8 thousand Walkman music players.")

for sent in doc.sents:
    displacy.render(
        nlp(sent.text),
        style = 'ent', 
        jupyter = True
    )

colors = {
    'ORG':'radial-gradient(yellow, green)'
    }
options = {
    'ents':['PRODUCT', 'ORG'],
    'colors':colors
    }

for sent in doc.sents:
    displacy.render(
        nlp(sent.text),
        style = 'ent', 
        jupyter = True,
        options = options
    )

# view on http://127.0.0.1:5000/
displacy.serve(
    doc, 
    style = 'ent',
    options = options
)