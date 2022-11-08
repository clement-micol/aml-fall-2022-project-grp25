from doc.core.fast_bow.fast_bow import text2bow, fast_encoding
from doc.core.logger import custom_logger
import pandas as pd
from tqdm import tqdm

class FastEncoding():
    def __init__(self,vocab_size=10_000,logger=None, verbose=False)->None:
        self.vocab_size = vocab_size
        self.verbose = verbose
        if logger == None:
            self.logger = custom_logger()
        else :
            self.logger = logger
    
    def adapt(self,data)->None:
        concatenated_sentences = [token for sentence in data for token in sentence]
        # We build the vocab out the bag of words
        if not self.verbose:
            self.logger.info("Starting to build the BoW")
        self.bow = text2bow(concatenated_sentences)
        self.bow = sorted(self.bow,key=lambda x: x[1],reverse=True)
        if not self.verbose:
            self.logger.info(f"Finished building the BoW : \n{self.bow[:100]}")
        self.bow = self.bow[:self.vocab_size]

    
    def __call__(self, data, output="int"):
        # Remove words not in the vocab
        concatenated_sentences = data.to_list()
        if not self.verbose:
            self.logger.info("Starting to encode text")
        encoded_data = fast_encoding(
            concatenated_sentences,
            self.bow,
            output,
            self.verbose
            )
        return encoded_data


if __name__ == "__main__":
    logger = custom_logger()
    logger.info("Loading the data")
    tweet_data = pd.read_parquet("./output/preprocessed_tweets.parquet")
    tweet_data = tweet_data["body_preprocessed"]
    encoder = FastEncoding(100_000,logger)
    encoder.adapt(tweet_data)
    print(encoder(tweet_data))