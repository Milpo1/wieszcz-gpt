import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


class TextDataset(Dataset):
    def __init__(self, data, block_size) -> None:
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data)-self.block_size-1
    def __getitem__(self, i):
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+self.block_size+1]      
        return x, y

@dataclass
class TrainConfig:
    max_iters = 5000
    eval_interval = 200
    eval_iters = 50
    batch_size = 16
    learning_rate = 5e-4
    num_workers=0

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, train_config=TrainConfig):
        assert train_dataset.__len__() > 0
        self.config = train_config
        self.train_dataloader = DataLoader(train_dataset,train_config.batch_size, shuffle=False, num_workers=self.config.num_workers, 
                                    sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)))
        self.val_dataloader = DataLoader(val_dataset,train_config.batch_size, shuffle=False, num_workers=self.config.num_workers, 
                                    sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=int(1e10))) \
                                        if val_dataset is not None else None
        self.model = model
        self.block_size = model.config.block_size
        self.device = model.device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.learning_rate)
    

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        config = self.config
        self.model.eval()
        loader_dict = {'train':self.train_dataloader}
        if self.val_dataloader is not None:
            loader_dict['val'] = self.val_dataloader
        
        for split, dataloader in loader_dict.items():
            losses = torch.zeros(config.eval_iters)
            data_iter = iter(dataloader)
            for k in range(config.eval_iters):
                X, Y = next(data_iter)
                logits, loss = self.model(X.to(self.device), Y.to(self.device))
                losses[k] = loss.item()
            out[split] = str(losses.mean().item())[:5]
        self.model.train()
        return out
    
    def train(self):
        self.model.train()
        config = self.config
        data_iter = iter(self.train_dataloader)
        for i in range(config.max_iters):
            if i % config.eval_interval == 0 or i == config.max_iters - 1:
                #print(f'{i} estimating loss')
                
                losses = self.estimate_loss()
                print(f"step {i} losses: {losses}")

            xb, yb = next(data_iter)

            logits, loss = self.model(xb.to(self.device), yb.to(self.device))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
    