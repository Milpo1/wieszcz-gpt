# WieszczGPT

A fully configurable GPT model & tokenizer made from scratch, with the architecture of [GPT 2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

Example usage can be found in the demo.py file or below in short:

```python
# %%
from tokenizer import Tokenizer
from model import WieszczGPT
from trainer import Trainer, TextDataset
import torch

with open('mickiewicz.txt', 'r') as file:
    raw_data = file.read()

model = WieszczGPT() # Default model config
model.to(model.device)
    
tok = Tokenizer(model.config.vocab_size)

data = tok.encode(raw_data)
train_dataset = TextDataset(data,model.config.block_size)

trainer = Trainer(model, train_dataset) # Default train config
trainer.train()

# Generate a few tokens with your newly trained GPT!
model.eval()
context = torch.zeros((1, model_config.block_size), dtype=torch.long, device=model.device)
gen = model.generate(context, max_new_tokens=2000)[0].tolist()
print(tok.decode(gen))

```
Possible output (quality depends on the size of the model):
```
Przebóg! Zasnąłem w duszę nie jagnet wybaczy. 
Ja bym imię poznasz. Już mi ranek: złamie ulgę zadzone!
I nie dostałam skrzywdziłem chłopczyna:
Lecz co idź tu, luba, tu nie gada!

GUSTAW
W domu i z innych wojskich Litworyjewa
Skacze, drugi i szpieg, i z tłumu nagle,
Gdy Bóg dotąd cały do brudzy przebacz hasło,
Gdyby ja nam gościom i zacznie zawierzeć pacierze.
Już dzieci wroga wpół wojene zagrał,
Nim bojów zacznie bywał się szlachtę właśnie:
Bo bracia stary en bez mnie bez szlachty i las podobnych
ano ruszach i tak karewicz na parkanych.
```
Shout out to Adam Mickiewicz and Andrej Karpathy.
