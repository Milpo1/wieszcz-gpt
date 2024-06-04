# %%
from tokenizer import Tokenizer
from model import WieszczGPT, GPTConfig
from trainer import Trainer, TextDataset, TrainConfig
import torch

with open('mickiewicz2.txt', 'r') as file:
    raw_data = file.read()
    
# %%
model_config = GPTConfig

tok = Tokenizer(model_config.vocab_size)
data = tok.encode(raw_data)
# %%
n = int(0.9*len(data))
train_dataset = TextDataset(data[:n],model_config.block_size)
val_dataset = TextDataset(data[n:],model_config.block_size)

model = WieszczGPT(GPTConfig)
model.to(model.device)

train_config = TrainConfig
trainer = Trainer(model,train_dataset, val_dataset,TrainConfig)
# %%
#RUNNN
trainer.train()
# %%
model.eval()
context = torch.zeros((1, model_config.block_size), dtype=torch.long, device=model.device)
#context = tok.encode('Litwo! Ojczyzno moja! ty jesteś jak zdrowie:\nIle cię trzeba cenić, ten tylko się dowie,\nKto cię stracił. Dziś piękność twą w całej ozdobie\nWidzę i opisuję, bo tęsknię po tobie.Panno święta, co Jasnej bronisz Częstochowy\nI w Ostrej świecisz Bramie! Ty, co gród zamkowy\nNowogródzki ochraniasz z jego wiernym ludem!')
#context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(dim=0)
gen = model.generate(context, max_new_tokens=2000)[0].tolist()
print(tok.decode(gen))
