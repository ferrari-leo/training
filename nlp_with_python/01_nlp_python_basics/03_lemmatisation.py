import spacy
nlp = spacy.load('en_core_web_sm')

def show_lemma(text):
    for token in text:
        print(f"{token.text:{12}}{token.lemma_:{12}}{token.pos:<{6}}{token.lemma:<{22}}")

doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today")
show_lemma(doc1)

doc2 = nlp(u"I saw ten mice today!")
show_lemma(doc2)