import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"The quick brown fox jumped over the lazy dog.")

displacy.render(
    doc, 
    style = 'dep',
    jupyter = True
    )

options = {
    'distance': 110,
    'compact': 'True',
    'color': 'yellow',
    'bg': '#09a3d5',
    'font': 'Times'
}

displacy.render(
    doc, 
    style = 'dep',
    jupyter = True,
    options = options
    )

doc2 = nlp(u"This is a sentence. This is another sentence, possibly longer than the other.")

spans = list(doc2.sents)
# View below at http://127.0.0.1:5000/
displacy.serve(
    spans,
    style = 'dep',
    options = {'distance':110}
)