import re
import spacy
import pandas as pd
from spacy.tokens import Doc
import string
from logger import custom_logger
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from tqdm.auto import tqdm
from joblib import Parallel

# To do : maybe shift to a Tensorflow preprocessing layer!

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

class FinancialTweetTokenizer:
    def __init__(self,lang):
        self.base_tokenizer = lang.tokenizer
        self.punct = string.punctuation
        self.punct = self.punct.replace("$","")
        self.punct = self.punct.replace("@","")
        self.punct = self.punct.replace("&","")
        prefixes = list(lang.Defaults.prefixes)
        prefixes.remove("\$")
        prefixes.append("-")
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        self.base_tokenizer.prefix_search = prefix_regex.search
        self.vocab = lang.vocab
        self.stop_words = lang.Defaults.stop_words
        self.stop_words.add("\w")
        if "up" in self.stop_words:
            self.stop_words.remove("up")
        if "down" in self.stop_words:
            self.stop_words.remove("down")
        if "top" in self.stop_words:
            self.stop_words.remove("top")

    def _is_removable(self,token):
        if token.lower_ in self.stop_words:
            return True
        elif token.is_punct or token.like_url:
            return True
        elif re.findall("\d+",token.text):
            return True
        elif "@" in token.text or "$" in token.text:
            return True
        elif len(token.text) == 1:
            return True
        else :
            return False

    def __call__(self,text):
        # First tokenize the text
        tokens = self.base_tokenizer(text)
        text_tokens = " ".join([token.lower_ for token in tokens if not self._is_removable(token)])
        tokens =  re.findall(r"[\w$@]+|["+self.punct+"]", text_tokens, re.UNICODE) 
        tokens = [token for token in tokens if not token in self.punct]
        return Doc(self.vocab,words=tokens)

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(texts,nlp):
        preproc_pipe = []
        for tokens in nlp.pipe(texts, batch_size=20):
            preproc_pipe.append([token.lemma_ for token in tokens])
        return preproc_pipe

def preprocess_parallel(texts, nlp, chunksize=100):
    executor = ProgressParallel(n_jobs=7, backend='multiprocessing', prefer="processes",total=len(texts)//chunksize)
    do = delayed(process_chunk)
    tasks = (do(chunk,nlp) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

if __name__=='__main__':
    blue = '\u001b[30m'
    logger = custom_logger()
    nlp = spacy.load("en_core_web_sm", disable=['parser','senter', 'ner'])
    nlp.tokenizer = FinancialTweetTokenizer(nlp)
    logger.info("Started loading the dataset")
    tweets_data = pd.read_csv("./data/Tweet.csv")
    logger.info("Finished loading the dataset")
    logger.info("Starting to preprocess")
    tweets_data["body_preprocessed"] = preprocess_parallel(tweets_data["body"],nlp,chunksize=10_000)
    logger.info("Finished preprocessing")
    tweets_data.to_parquet("./output/preprocessed_tweets.parquet")
    logger.info("Saved the preprocessed file to"+blue+" './ouput/preprocessed_tweets.parquet'")