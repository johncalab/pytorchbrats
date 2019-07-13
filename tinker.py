"""
Stolen from here:
http://andrewjmoodie.com/2018/03/python-3-silly-random-name-generator/
"""
import urllib.request
import random

word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
response = urllib.request.urlopen(word_url)
long_txt = response.read().decode()
words = long_txt.splitlines()

upper_words = [word for word in words if word[0].isupper()]
name_words  = [word for word in upper_words if not word.isupper()]
one_name = ' '.join([name_words[random.randint(0, len(name_words))] for i in range(2)])


def randoname():
   name = ' '.join([name_words[random.randint(0, len(name_words))] for i in range(2)])
   return name