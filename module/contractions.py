import re

def contractions_list():
    contractions = {
        "general": {
            "n't": " not",
            "'re": " are",
            "'s": "",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
        },
        "specific": {
            "ain't": "is not",
            "can't": "can not",
            "'cause": "because",
            "he's": "he is",
            "how's": "how is",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "o'clock": "of the clock",
            "shan't": "shall not",
            "she's": "she is",
            "so's": "so is",
            "that's": "that is",
            "there's": "there is",
            "what's": "what is",
            "when's": "when is",
            "where's": "where has / where is",
            "who's": "who is",
            "why's": "why is",
            "won't": "will not",
            "y'all": "you all",
        }
    }
    return contractions

def expand_contractions(data_x):
    contractions = contractions_list()

    #firstly expand specific contractions
    specific_contractions = contractions['specific']
    pattern = re.compile('|'.join(specific_contractions.keys()))
    result = [pattern.sub(lambda x: specific_contractions[x.group()], comment) for comment in data_x]

    #secondly expand general contractions
    general_contractions = contractions['general']
    pattern = re.compile('|'.join(general_contractions.keys()))
    result = [pattern.sub(lambda x: general_contractions[x.group()], comment) for comment in result]
    return result