import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language = 'english')
words = ['run', 'runner', 'ran', 'runs', 'easily', 'fairly', 'fairness']
for word in words: print(f'{word} ----> {p_stemmer.stem(word)}')
print('\n')
for word in words: print(f'{word} ----> {s_stemmer.stem(word)}')

words = ['generous', 'generation', 'generously', 'generate']
for word in words: print(f'{word:>{20}} ----> {p_stemmer.stem(word):{20}}')
print('\n')
for word in words: print(f'{word:>{20}} ----> {s_stemmer.stem(word):{20}}')