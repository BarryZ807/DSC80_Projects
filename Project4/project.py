# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with
        standard new-lines (\n).
        - If the function is called twice in succession, it should not
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    request = requests.get(url)
    time.sleep(5)
    text = request.text
    start = "\*\*\* START[\w\s]+PROJECT GUTENBERG EBOOK[\w\s]+ \*\*\*"
    end = "\*\*\* END[\w\s]+PROJECT GUTENBERG EBOOK[\w\s]+ \*\*\*"
    content = re.split(start, re.split(end, text)[0])[1]
    transformed = re.sub('\r\n', '\n', content)
    return transformed


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens
    satisfying the following conditions:
        - The start of any paragraph should be represented in the
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    token_lst = []
    start = '\x02'
    end = ' \x03'
    combined = ' \x03\n\x02 '

    start_break = '^(\s*)'
    end_break = '(\s+)$'
    combined_break = '(\s\s+)'


    format_start_break = re.sub(start_break, start, book_string)
    format_end_break = re.sub(end_break, end, format_start_break)
    formatted = re.sub(combined_break, combined, format_end_break)
    split = re.split(r'\b|\s', formatted)
    for token in split:
        if token != ' ' and token != '':
            if re.match('\W+', token) != None:
                token_lst.extend(list(token))
            else:
                token_lst.append(token)

    return token_lst


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        n_unique_tokens = len(set(tokens))
        probs = 1 / n_unique_tokens
        return pd.Series(probs, index=np.unique(tokens))

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        try:
            prob = 1
            for word in words:
                prob *= self.mdl[word]
            return prob
        except:
            return 0

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        return ' '.join(np.random.choice(self.mdl.index, size=M, replace=True))



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):

    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        return pd.Series(tokens).value_counts(normalize=True)

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        try:
            prob = 1
            for word in words:
                prob *= self.mdl[word]
            return prob
        except: # look into what type of error should go here
            return 0

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        return ' '.join(np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl))



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):

    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            # self.prev_mdl = mdl.prev_mdl
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams.
        The START/STOP tokens in the N-grams should be handled as
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        ngram_lst = []
        for i in range(len(tokens)-self.N+1):
            ngram_lst.append(tuple([tokens[i+j] for j in range(self.N)]))
        return ngram_lst

    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        if len(ngrams) == 0:
            return pd.DataFrame()
        # ngram counts C(w_1, ..., w_n)
        unique_ngrams_count = pd.Series(ngrams).value_counts()
        unique_ngrams = unique_ngrams_count.index.to_series()
        # n-1 gram counts C(w_1, ..., w_(n-1))
        unique_n1grams_count = pd.Series([ngram[:-1] for ngram in ngrams]).value_counts()

        df = pd.DataFrame({
            'ngram': unique_ngrams,
            'n1gram': unique_ngrams.apply(lambda x: x[:-1]),
            })

        # Create the conditional probabilities
        probability = df.apply(
            lambda x: unique_ngrams_count[x['ngram']]/unique_n1grams_count[x['n1gram']], axis=1)


        # Put it all together
        df = df.assign(prob=probability).reset_index(drop=True)

        return df
        ...
        ...

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        ngrams = pd.DataFrame({'ngram':self.create_ngrams(words)})
        merged_df = pd.merge(ngrams, self.mdl, how='left', on='ngram').fillna(0)
        prob = merged_df['prob'].prod()

        return prob * self.prev_mdl.probability(words[:self.N-1])


    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        # Use a helper function to generate sample tokens of length `length`
        def get_tokens(self, M):
            if M == 0:
                word = ('\x02',)
            else:
                if M <= self.N-1:
                    prev_tokens = get_tokens(self.prev_mdl, M-1)

                    word_lst = self.mdl[self.mdl['n1gram']==prev_tokens]
                    if word_lst.shape[0] == 0:
                        word = prev_tokens + tuple('\x03')
                    else:
                        word = np.random.choice(word_lst['ngram'], size=1, replace=True, p=word_lst['prob'])[0]
                else:
                    prev_tokens = get_tokens(self, M-1)
                    word_lst = self.mdl[self.mdl['n1gram']==prev_tokens[-self.N+1:]]
                    if word_lst.shape[0]== 0:
                        word = prev_tokens + tuple('\x03')
                    else:
                        word = np.random.choice(word_lst['ngram'], size=1, replace=True, p=word_lst['prob'])[0]
                        word = prev_tokens[:-self.N+1] + word
            return word

        tokens = get_tokens(self, M)
        # Transform the tokens to strings
        string = ' '.join(list(tokens))
        return string
