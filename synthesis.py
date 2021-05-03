import nltk
import pyphen
import pyttsx3

arpabet = nltk.corpus.cmudict.dict()
engine = pyttsx3.init()
dic = pyphen.Pyphen(lang='en_GB')
sentence = "This is a pure Python module to hyphenate text using existing Hunspell hyphenation dictionaries"
words = sentence.split(" ")
syllable_words = []
for word in words:
    word = word.lower()
    word_phonemes = arpabet[word]
    word_syllables = dic.inserted(word).split("-")
    syllable_words += word_syllables

for syllable_word in syllable_words:
    for syllable in syllable_word:
        # syllable = syllable.replace("y", "i")
        engine.say(syllable)

engine.runAndWait()