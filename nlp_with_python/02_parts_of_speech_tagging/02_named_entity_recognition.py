import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
nlp = spacy.load('en_core_web_sm')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(f"{ent.text} - {ent.label_} - {str(spacy.explain(ent.label_))}")
    else:
        print("No entities found")

#region Part 1      
doc = nlp(u"Hi how are you?")
show_ents(doc)

doc = nlp(u"May I go to Washington, DC next May to see the Washington Monument?")
show_ents(doc)

doc = nlp("Can I please have 500 dollars of Microsoft stock?")
show_ents(doc)

doc = nlp(u"Tesla to build a U.K. factoy for $6 million")
show_ents(doc)

# add a new entitiy
ORG = doc.vocab.strings[u"ORG"]
new_ent = Span(doc, 0, 1, label=ORG)
doc.ents = list(doc.ents) + [new_ent]
show_ents(doc)
#endregion
#region Part 2
# add multiple new entities
doc = nlp(u"Our company created a brand new vacuum cleaner."
          u"This new vacuum-cleaner is the best in show.")
show_ents(doc)

matcher = PhraseMatcher(nlp.vocab)
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]
matcher.add('newproduct', None, *phrase_patterns)

found_matches = matcher(doc)
print(found_matches)

PROD = doc.vocab.strings[u"PRODUCT"]
new_ents = [Span(doc, match[1], match[2], label = PROD) for match in found_matches]

doc.ents = list(doc.ents) + new_ents
show_ents(doc)

doc = nlp(u"riginally I paid $29.95 for this car toy, but now it is marked down by 10 dollars.")
len([ent for ent in doc.ents if ent.label_ == "MONEY"])

#endregion