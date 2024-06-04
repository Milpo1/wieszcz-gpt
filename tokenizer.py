import pandas as pd
import torch
BASE_CHAR = 256
class Tokenizer():
    def __init__(self, vocab_size) -> None:
        self.vocab_size = max(BASE_CHAR,vocab_size)
        self.merges = {}
        
    #Get frequency count for each pair of tokens in text
    def get_pair_counts(self, text):
        pairs = {}
        for i in range(len(text)-1):
            pair = text[i],text[i+1]
            pairs[pair] = pairs.get(pair,0)+1
        return pairs

    #Replace pairs in the text with a new token
    def merge_pair(self, text, merge_pair, new_token):
        new_text = []
        i=0
        while i < len(text):
            pair = text[i], -1 if i+1 >= len(text) else text[i+1]
            
            if pair == merge_pair:
                new_text.append(new_token)
                i+=2
            else:
                new_text.append(text[i])
                i+=1
        return new_text

    #Encode bytes into tokens using vocab of size up to vocab size.
    def encode(self, text):
        if type(text) == type(''):
            text = [c for c in text.encode('utf-8')]
        if type(text) == type(torch.tensor([])):
            text = text.tolist()
        max_token= max(BASE_CHAR,max(text))
        merges_number = self.vocab_size - max_token
        assert merges_number >= 0, 'Maximum token greater than vocab size'
        
        for pair, token in self.merges.items():
            text = self.merge_pair(text,pair, token)
        
        while len(text) > 1 and len(self.merges) < merges_number:
            pairs = self.get_pair_counts(text)
            new_token = BASE_CHAR + len(self.merges)
            merge_pair = max(pairs,key=pairs.get)
            self.merges[merge_pair] = new_token
            text = self.merge_pair(text,merge_pair,new_token)
        return torch.tensor(text)

    #Decode tokens into bytes
    def decode(self, text):
        if type(text) == type(''):
            text = [c for c in text.encode('utf-8')]     
        if type(text) == type(torch.tensor([])):
            text = text.tolist()   
        for pair, token in reversed(self.merges.items()):
            i = 0
            new_text = []
            while i < len(text):
                if text[i] == token:
                    new_text.extend(pair)
                else:
                    new_text.append(text[i])
                i+=1
            text = new_text
        return bytes(text).decode('utf-8', errors='replace')