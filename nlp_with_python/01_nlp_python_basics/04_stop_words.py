import spacy
nlp = spacy.load('en_core_web_sm')

# stop words are filler words in a language that don't add additional meaning to a sentence
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words)) # returns 305
print(nlp.vocab['mystery'].is_stop) # check if a word is a stop word

# add new stop words
nlp.Defaults.stop_words.add('btw')
nlp.vocab['btw'].is_stop = True
print(len(nlp.Defaults.stop_words)) # returns 306

# remove stop words in original set
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyond'].is_stop = False
print(len(nlp.Defaults.stop_words)) # returns 305
