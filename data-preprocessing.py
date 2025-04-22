"""
Steps for Data Preparation:
1. Cleanup: Remove html, emoji handler and spelling corrections
2. Basic Preprocessing
    1. Tokenization: 
        1. Sentence tokenization: When you tokenize the whole sentence into a number/token.
        2. Word tokenization: When you perform tokenization on each word of a sentence and it is converted to numbers/tokens (mostly used).
    2. Remove stop words: we can analyze the context/sentiment of the sentence without the stop words. Removing them will also reduce dimensionality in the vectors.
    3. Steaming (less used): When there are different tenses/verb forms of the same root word (play, played, playing), we just consider the root word (play). It will help reduce dimension when converting the words to vectors.
    Note: When there are more dimensions, the model tends to get confuse which is known as Curse of Dimensionality.
    4. Lemmatization (frequently used): Similar to steaming but the the root word is readable whereas the root word is not readable in steaming.
    5. Punctuation removal
    6. Lower case: In 2 given sentences having 2 same words but with different cases (Eg: Ria is a girl. ria is a data scientist), the model considers it to be different words due to ASCII interpretation. Hence, before feeding it to the model, we make all text lower case to avoid model to get confused. 
    7. Language detection
3. Advanced Preprocessing
    1. Parts of speech tagging: Tagging every word/token as parts of speech like noun, verb, adjective, etc.
    2. Parsing
    3. Coreference resolution: When the subject of the sentence is again referred in another part of the sentence, the model should be capable to understand (as shown in image)

"""

#Importing required packages
import pandas as pd

#Location of the csv file
dataPath = "" #Path to dataset

#Load the data
df = pd.read_csv(dataPath)

#Shape of data
print(df.shape)

#Number of columns in the csv
print(df.head())

#Considering only the first 100 rows for computing purposes
df = df.head(100)
print(df.shape)


#Converting Lowercase to reduce dimensionality
df['review'] = df['review'].str.lower()


#Remove HTML Tags
import re
def removeHtmlTags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)
text = "<html><body><p> Movie 1</p><p> Actor - Aamir Khan</p><p> Click here to <a href='http://google.com'>download</a></p></body></html>"
print(removeHtmlTags(text))

df['review'] = df['review'].apply(removeHtmlTags)


#Remove URL
def removeUrl(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

text1 = 'Check out my youtube https://www.youtube.com/dswithbappy dswithbappy'
print(removeUrl(text1))


#Punctuation Handling
import string
exclude = string.punctuation
def removePunc(text):
    return text.translate(str.maketrans('', '', exclude))
text2 = 'string. With, Punctuation?'
print(removePunc(text2))

df['review'] = df['review'].apply(removePunc)


#Handle Chat Conversation (Abbreviations)
chatAbbrev = {
    'AFAIK':'As Far As I Know',
    'AFK':'Away From Keyboard',
    'ASAP':'As Soon As Possible',
    "FYI": "For Your Information",
    "ASAP": "As Soon As Possible",
    "BRB": "Be Right Back",
    "BTW": "By The Way",
    "OMG": "Oh My God",
    "IMO": "In My Opinion",
    "LOL": "Laugh Out Loud",
    "TTYL": "Talk To You Later",
    "GTG": "Got To Go",
    "TTYT": "Talk To You Tomorrow",
    "IDK": "I Don't Know",
    "TMI": "Too Much Information",
    "IMHO": "In My Humble Opinion",
    "ICYMI": "In Case You Missed It",
    "AFAIK": "As Far As I Know",
    "BTW": "By The Way",
    "FAQ": "Frequently Asked Questions",
    "TGIF": "Thank God It's Friday",
    "FYA": "For Your Action",
    "ICYMI": "In Case You Missed It",
}

def chatConversation(text):
    newText = []
    for w in text.split():
        if w.upper() in chatAbbrev:
            newText.append(chatAbbrev[w.upper()])
        else:
            newText.append(w)
    return " ".join(newText)
print(chatConversation('Do this ASAP'))


#Handling Incorrect Text
from textblob import TextBlob
incorrectText = 'ccertain confitionas duriing seveal ggeneratuions aree modifiedds in thhe sammee mannner'
textBlb = TextBlob(incorrectText)
print(textBlb.correct().string)


#Remove Stopwords
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

def removeStopwords(text):
    newText = []
    for word in text.split():
        if word in stopwords.words('english'):
            newText.append('')
        else:
            newText.append(word)
    x = newText[:]
    newText.clear()
    return " ".join(x)

print(removeStopwords('probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it\'s not preachy or boring. it just never gets old, despite my having seen it some 15 or more times'))

print(df['review'].apply(removeStopwords))


#Handle Emojis
import re
def removeEmoji(text):
    emojiPattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    return emojiPattern.sub(r'', text)

print(removeEmoji("LMAO 😂"))


#Emojis are being used to understand pretext in chatGPT and hence they are necessary.
#Install the "emoji" package (pip install emoji) that will give us the meaning of every emoji.
import emoji
print(emoji.demojize("Python is 🔥"))


#Tokenization
#Using split() function
#Word tokenization
sentence1 = 'I am going to Delhi'
print(sentence1.split())

#Sentence tokenization
sentence2 = "I am going to Delhi. I will stay there for 3 days. Let's hope the trip will be great!"
print(sentence2.split('.'))

#Problem with split() function - it will pass the regular expression too along with it or
#will not be able to parse sentences correctly when more than 2 sentences are there
sentence3 = 'I am going to Delhi!'
sentence4 = 'Where do you think I should go? I have holidays for 3 days.'
print(sentence3.split()) #Word-level tokenization
print(sentence4.split('.')) #Sentence-level tokenization

#Tokenization using Regular Expression
import re
words = re.findall("[\w']+", sentence3)
print(words)
sentences = re.compile('[.!?]').split(sentence4)
print(sentences)

#Tokenization using Natural Language Toolkit
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt_tab') #This is used to perform tokenization

print(word_tokenize(sentence3))
print(sent_tokenize(sentence4))

#Spacy
import spacy
nlp = spacy.load('en_core_web_sm')
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)
doc3 = nlp(sentence3)
doc4 = nlp(sentence4)

for token in doc4:
    print(token)


#Stemmer (getting root word)- not readable since sometimes the word is converted to such a form which only
#useful for the model to understand and not very understandable for humans to read
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def stemWords(text):
    return " ".join([ps.stem(word) for word in text.split()])

sample = "walk walks walking walked"
print(stemWords(sample))

#Example of non-readable function
text = 'probably my alltime favorite movie a story of selflessness sacrifice and dedication to a noble cause but its not preachy or boring it just never gets old despite my having seen it some 15 or more times in the last 25 years paul lukas performance brings tears to my eyes and bette davis in one of her very few truly sympathetic roles is a delight the kids are as grandma says more like dressedup midgets than children but that only makes them more fun to watch and the mothers slow awakening to whats happening in the world and under her own roof is believable and startling if i had a dozen thumbs theyd all be up for this movie'
print(text)

print(stemWords(text))


#Lemmatization - Gives readable words
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
wordNetLemmatizer = WordNetLemmatizer()

sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
punctuations = "?:!.,;"
sentenceWords = nltk.word_tokenize(sentence)
for word in sentenceWords:
    if word in punctuations:
        sentenceWords.remove(word)

print(sentenceWords)
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentenceWords:
    print ("{0:20}{1:20}".format(word,wordNetLemmatizer.lemmatize(word,pos='v')))

#Note: Stemming & lemmatization have the same funcationality to retrieve root words
# but lamatization is works better. Lemmatization is slow & stemming is fast since 
# lemmatization gives words in readable format.