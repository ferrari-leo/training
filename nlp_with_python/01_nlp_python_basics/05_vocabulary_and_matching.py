# https://spacy.io/usage/rule-based-matching

import spacy
nlp = spacy.load('en_core_web_sm')
#region Part 1
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# SolarPower
pattern1 = [{'LOWER':'solarpower'}]
# Solar-Power
pattern2 = [{'LOWER':'solar'}, {'IS_PUNCT':True}, {'LOWER':'power'}]
# Solar Power
pattern3 = [{'LOWER':'solar'}, {'LOWER':'power'}]

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

doc = nlp(u"The Solar Power industry continues to grow as solarpower increases. Solar-power is amazing")

found_matches = matcher(doc)
print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)
    
matcher.remove('SolarPower')

# now introduce quantifiers to OP key
pattern1 = [{'LOWER':'solarpower'}]
pattern2 = [{'LOWER':'solar'}, {'IS_PUNCT':True, 'OP':'*'}, {'LOWER':'power'}]
matcher.add('SolarPower', None, pattern1, pattern2)

doc2 = nlp(u"Solar--power is solarpower")

found_matches = matcher(doc2)
print(found_matches)
#endregion
#region Part 2
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

with open("../TextFiles/reaganomics.txt", encoding = 'unicode_escape') as f:
    doc3 = nlp(f.read())
    
phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']

phrase_patterns = [nlp(text) for text in phrase_list]

matcher.add('EconMatcher', None, *phrase_patterns)

found_matches = matcher(doc3)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc3[start-5:end+5]
    print(match_id, string_id, start, end, span.text)

#endregion