import spacy
from spacy.pipeline import SentenceSegmenter
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"This is the first sentence. This is another sentence. This is the last sentence.")

for sent in doc.sents: print(sent)

doc = nlp(u'"Management is doing the right things; leadership is doing the right things." - Peter Drucker')

for sent in doc.sents:
    print(sent)
    print('\n')
    
# Add segmentation rule
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe(set_custom_boundaries, before = 'parser')

doc4 = nlp(u'"Management is doing the right things; leadership is doing the right things." - Peter Drucker')
for sent in doc4.sents:
    print(sent)
# Change segmentation rule
nlp = spacy.load('en_core_web_sm')
mystring =u"This is a sentence. This is another. \n\nThis is a \nthird sentence."
doc = nlp(mystring)
for sent in doc.sents:
    print(sent)
    
# Want new line break to be sentence segmentation
def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'):
            seen_newline = True
    yield doc[start:]
    
sbd = SentenceSegmenter(nlp.vocab, strategy = split_on_newlines)
nlp.add_pipe(sbd)

doc = nlp(mystring)
for sent in doc.sents:
    print(sent)