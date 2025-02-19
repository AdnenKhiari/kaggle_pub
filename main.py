# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

train_set = pd.read_csv('/kaggle/input/the-bards-best-a-character-modeling-dataset/train.csv').loc[0,'text']
valid_set = pd.read_csv('/kaggle/input/the-bards-best-a-character-modeling-dataset/validation.csv').loc[0,'text']
test_set = pd.read_csv('/kaggle/input/the-bards-best-a-character-modeling-dataset/test.csv').loc[0,'text']

class Tokenizer:
    def __init__(self,text):
        self.vocab = list(set(self.tokenize(text)))
        self.encod_dict = { item:index  for index,item in enumerate(self.vocab)  }        
        self.decode_dict = { index:item  for index,item in enumerate(self.vocab)  }
    def vocab_size(self):
        return len(self.vocab)
    # def tokenize(self,text):
    #     t = text.split(' ')
    #     res = []
    #     for item in t:
    #         res.append(item)
    #         res.append(' ')
    #     return res
    def tokenize(self, text):
        return [text[i:i+1] for i in range(len(text))]

    def encode(self,text):
        return [self.encod_dict[item] for item in self.tokenize(text)]
    def decode(self,tokens):
        return ''.join([self.decode_dict[item] for item in tokens])
    

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
class DataShak(Dataset):
    def __init__(self,tokenizer,text,context_size):
        self.text = tokenizer.encode(text)
        self.context_size = context_size
        print('DataSet Contains',len(self.text),"Tokens",len(self.text) // self.context_size,"Sequences")
    def __len__(self):
        return len(self.text) // self.context_size
    def __getitem__(self,idx):
        next_idx = idx+1
        x = self.text[idx*self.context_size:next_idx*self.context_size]
        y = self.text[idx*self.context_size+1:next_idx*self.context_size+1]
        if len(y) < self.context_size:
            y = self.text[-self.context_size+1:] + [' ']
        if len(x) < self.context_size:
            x = self.text[-self.context_size:]
        return x,y
    
def collate_fn(batch):
    xs = []
    ys = []
    for x,y in batch:
        xs.append(torch.tensor(x))
        ys.append(torch.tensor(y))
    return torch.stack(xs),torch.stack(ys)
        

# class SimpleAverage(torch.nn.Module):
#     def __init__(self,embed_dim,head_dim):
#         super().__init__()
#         self.value_projection = torch.nn.Sequential(torch.nn.Linear(embed_dim,head_dim))
#         # self.key_projection = torch.nn.Sequential(torch.nn.Linear(embed_dim,head_dim))
#         # self.queries_projection = torch.nn.Sequential(torch.nn.Linear(embed_dim,head_dim))
#     def forward(self,x):
#         tria = torch.tril(torch.ones(x.size(1),x.size(1)))
#         avg_matrix = torch.zeros(x.size(1),x.size(1))
#         avg_matrix = torch.nn.functional.softmax(avg_matrix.masked_fill(tria == 0,float('-inf')),-1).to(DEVICE)

#         value = self.value_projection(x)
#         mixed_value = avg_matrix @ value 
        
#         return mixed_value
class CausalSelfAttention(torch.nn.Module):
    def __init__(self,embed_dim,head_dim):
        super().__init__()
        self.value_projection = torch.nn.Linear(embed_dim,head_dim)
        self.key_projection = torch.nn.Linear(embed_dim,head_dim)
        self.queries_projection = torch.nn.Linear(embed_dim,head_dim)
        self.norm = head_dim**0.5
    def forward(self,x):
        tria = torch.tril(torch.ones(x.size(1),x.size(1))).to(DEVICE)
        queries = self.queries_projection(x)
        keys = self.key_projection(x)
        att_matrix = queries @ keys.transpose(-2,-1) / self.norm
        att_matrix = torch.nn.functional.softmax(att_matrix.masked_fill(tria == 0,float('-inf')),-1)

        value = self.value_projection(x)
        mixed_value = att_matrix @ value
        
        return mixed_value
        

class MultiHeadedCausalSelfFlashAttention(torch.nn.Module):
    def __init__(self,embed_dim,head_dim,num_heads):
        super().__init__()
        assert head_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_projection = torch.nn.Linear(embed_dim,head_dim) 
        self.key_projection = torch.nn.Linear(embed_dim,head_dim)
        self.queries_projection = torch.nn.Linear(embed_dim,head_dim)
        self.norm = (head_dim//num_heads)**0.5
    def forward(self,x):
        queries = self.queries_projection(x)  # ( B , C , D * N) N = Number of heads , C = Context Length , D = Head Embed Dim
        keys = self.key_projection(x) # ( B , C , D * N)
        value = self.value_projection(x) # ( B , C , D * N)

        queries = queries.reshape(x.size(0),x.size(1),self.num_heads,self.head_dim // self.num_heads).transpose(-3,-2) # ( B , N, C , D )
        keys = keys.reshape(x.size(0),x.size(1),self.num_heads,self.head_dim // self.num_heads).transpose(-3,-2) # ( B , N, C , D )
        value = value.reshape(x.size(0),x.size(1),self.num_heads,self.head_dim // self.num_heads).transpose(-3,-2) # ( B , N, C , D )
        

        mixed_value = torch.nn.functional.scaled_dot_product_attention(queries,keys,value,is_causal=True) # ( B , N, C , D )
        mixed_value = mixed_value.transpose(-3,-2).reshape((x.size(0),x.size(1),self.head_dim))
        
        return mixed_value
        
class GPTMLP(torch.nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(embed_dim)
        self.fc = torch.nn.Linear(embed_dim,4*embed_dim)
        self.actv = torch.nn.GELU()
        self.proj = torch.nn.Linear(4*embed_dim,embed_dim)
        self.proj.FIX_INIT = 1
    def forward(self,x):
        x = self.actv(self.fc(self.ln_1(x)))
        x = self.proj(x)
        return x

class TransformerBlock(torch.nn.Module):
    def __init__(self,embed_dim,head_dim,num_heads):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(embed_dim)
        self.attn = MultiHeadedCausalSelfFlashAttention(embed_dim,head_dim,num_heads)
        self.proj = torch.nn.Linear(head_dim,embed_dim)
        self.proj.FIX_INIT = 1

        self.mlp = GPTMLP(embed_dim)
    def forward(self,x):
        x = x + self.proj(self.attn(self.ln_1(x)))
        x = x + self.mlp(x)
        return x
    
class GPT(torch.nn.Module):
    def __init__(self,vocab_size,context_size,embed_dim,head_dim,n_heads,n_layers):
        super().__init__()
        self.n_layers = n_layers

        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(vocab_size,embed_dim),
            wpe = torch.nn.Embedding(context_size,embed_dim),
            drop = torch.nn.Dropout(0.1),
            h = torch.nn.ModuleList(
                [TransformerBlock(embed_dim,head_dim,n_heads) for _ in range(n_layers)]
            )
        ))

        self.lm_head = torch.nn.Linear(embed_dim,vocab_size)
        self.register_buffer('wpe_sequence',torch.arange(0,context_size))
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module,'FIX_INIT'):
                module.weight.data *= (2/self.n_layers)**0.5
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,x):
        x = self.transformer.wte(x)
        x = x + self.transformer.wpe(self.wpe_sequence[:x.size(1)])
        x = self.transformer.drop(x)

        # Thinking Phase
        for block in self.transformer.h:
            x = block(x)

        #Output
        x = self.lm_head(x)
        return x
    
# check vid if i need to divide loss to fix gradients sum
# only executed by worker 0
def validate(model,val_loader,loss_fn,device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (train_x, train_y) in enumerate(val_loader):
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                train_x = train_x.to(device)  # (B,T)
                train_y = train_y.to(device)  # (B,T)
        
                pred_x = model(train_x)  # (B,T,VOCAB_SIZE)
                loss = loss_fn(pred_x.view(-1, pred_x.size(-1)), train_y.view(-1))   # Flatten for CE loss
                total_loss += loss.item() 
     
        return total_loss / len(val_loader) 
            
def train(model, epoch,mini_batch_size,total_batch_size, train_loader,val_loader,optim,scheduler,loss_fn,device):
    model.train()

    print(len(train_loader.dataset),total_batch_size,f"{len(train_loader) / total_batch_size} Steps in an epoch")

    grad_acc_steps = total_batch_size / mini_batch_size
    scaler = torch.amp.GradScaler()

    for e in range(epoch):
        start_time = time.perf_counter()

        total_loss = 0
        token_th = 0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            print(batch_idx,train_x.shape)
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                train_x = train_x.to(device)  # (B,T)
                train_y = train_y.to(device)  # (B,T)
        
                pred_x = model(train_x)  # (B,T,VOCAB_SIZE)
                loss = loss_fn(pred_x.view(-1, pred_x.size(-1)), train_y.view(-1))   # Flatten for CE loss
                # Fix Grad Acc Averaging
                loss = loss / grad_acc_steps 

                # Report Statistics
                with torch.no_grad():
                    total_loss += loss.item() 
                    token_th = token_th + train_x.size(0) * train_x.size(1)
            
            # All Reduce
            scaler.scale(loss).backward()

            # Gradient Accumulation
            if (batch_idx) % grad_acc_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad()  # Clear previous gradients
    
        # Compute Statistics
        avg_loss = total_loss / len(train_loader)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        token_th = token_th / execution_time

        # Get Val Loss
        avg_val_loss = validate(model,val_loader,loss_fn,device)

        print(f"{device} Epoch {e} completed. Average Loss: {avg_loss:.4f} | Average Validation Loss: {avg_val_loss:.4f} |  Execution Time: {execution_time: .5f} s | Tokens Throughput : {token_th: .5f} t/s")


class AutoRegressiveGenerator():
    def __init__(self,tokenizer,model,context_size,device):
        self.tokenizer = tokenizer
        self.model = model
        self.context_size = context_size
        self.device=device
    def generate(self,inp,max_tokens,temperature,top_k=20):
        encoded = torch.tensor(self.tokenizer.encode(inp),device=self.device).view(1,-1) # (B,T)
        x = encoded
        result = x
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                x = x[:,-self.context_size:]
                next_token_distribution =  torch.nn.functional.softmax(self.model(x)[:,-1,:] / temperature,-1)
                # next_token = next_token_distribution.argmax(1).view(-1,1) # (B,1)
                # next_token = torch.multinomial(torch.topk(next_token_distribution,top_k,dim=1).values,1)
                next_token = torch.multinomial(next_token_distribution,1)
                x = torch.cat((x,next_token),1) # (B,T+1)
                result = torch.cat((result,next_token),1) # (B,T+1)
            return self.tokenizer.decode(result.tolist()[0])
        

from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group,destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import os
def ddp_setup(rank,world_size):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)


torch.set_float32_matmul_precision('high')

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
def main(rank, cfg):
    try:
        ddp_setup(rank, cfg["WORLD_SIZE"])
        DEVICE = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        tokenizer = Tokenizer(train_set)
        train_data = DataShak(tokenizer, train_set, cfg["CONTEXT_SIZE"])
        val_data = DataShak(tokenizer, valid_set, cfg["CONTEXT_SIZE"])
        train_loader = DataLoader(
            train_data,
            batch_size=cfg["TOTAL_WORKERS_BATCH_SIZE"],
            collate_fn=collate_fn,
            shuffle=False,
            sampler=DistributedSampler(train_data)
        )
        val_loader =  DataLoader(
            val_data,
            batch_size=cfg["WORKER_BATCH_SIZE"],
            collate_fn=collate_fn,
            shuffle=False
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        print("Vocab Size",tokenizer.vocab_size())
        model = GPT(
            tokenizer.vocab_size(),
            cfg["CONTEXT_SIZE"],
            cfg["EMBED_DIM"],
            cfg["EMBED_DIM"],
            cfg["NUM_HEADS"],
            cfg["NUM_LAYERS"]
        ).to(DEVICE)
        
        optim = torch.optim.AdamW(model.parameters(), lr=cfg["MIN_LR"],weight_decay=cfg["WEIGHT_DECAY"],fused=True)  # Instantiate optimizer

        scheduler = CosineAnnealingWarmupRestarts(optim,
                                          first_cycle_steps=cfg['CYCLE_STEPS'],
                                          cycle_mult=cfg['CYCLE_MULT'],
                                          max_lr=cfg["MAX_LR"],
                                          min_lr=cfg["MIN_LR"],
                                          warmup_steps=cfg['WARMUP_STEPS'],
                                          gamma=cfg['GAMMA'])

        model = DistributedDataParallel(model)
        train(
            model,
            cfg["EPOCH"],
            cfg["WORKER_BATCH_SIZE"],
            cfg["WORKER_GRAD_ACCUMULATION_BATCH_SIZE"],
            train_loader,
            val_loader,
            optim,
            scheduler,
            loss_fn,
            DEVICE
        )
        
        gen = AutoRegressiveGenerator(tokenizer, model, cfg["CONTEXT_SIZE"], DEVICE)
        print(gen.generate("As shall with either part", 200, 0.05))
    
    finally:
        destroy_process_group()
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training Configuration")
    parser.add_argument("--context_size", type=int, default=256, help="Context size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--epoch", type=int, default=2, help="Number of epochs per run")
    parser.add_argument("--num_epochs", type=int, default=1500, help="Total number of training epochs")
    parser.add_argument("--total_batch_size", type=int, default=2048, help="Total batch size across workers")
    parser.add_argument("--worker_batch_size", type=int, default=512, help="Batch size per worker")
    parser.add_argument("--max_lr", type=float, default=0.005, help="Max Learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0001, help="Min Learning rate")
    parser.add_argument("--gamma", type=float, default=0.05, help="Reduction of max lr of scheduler")
    parser.add_argument("--warmup_steps", type=float, default=2000, help="No of warmup steps")
    parser.add_argument("--cycle_steps", type=float, default=8000, help="No of steps per cycle")
    parser.add_argument("--cycle_mult", type=float, default=0.2, help="cycle annealing duration")
    parser.add_argument("--weight_decay", type=float, default=0.08, help="Weight Decay of AdamW")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    config = {
        "WORLD_SIZE": torch.cuda.device_count(),
        "CONTEXT_SIZE": args.context_size,
        "EMBED_DIM": args.embed_dim,
        "NUM_HEADS": args.num_heads,
        "EPOCH": args.epoch,
        "NUM_LAYERS": args.num_layers,
        "TOTAL_BATCH_SIZE": args.total_batch_size,
        "WORKER_BATCH_SIZE": args.worker_batch_size,
        "TOTAL_WORKERS_BATCH_SIZE": args.worker_batch_size * torch.cuda.device_count(),
        "NUM_EPOCHS": args.num_epochs,
        "MAX_LR": args.max_lr,
        "MIN_LR": args.min_lr,
        "GAMMA": args.gamma,
        "WARMUP_STEPS": args.warmup_steps,
        "CYCLE_STEPS": args.cycle_steps,
        "CYCLE_MULT": args.cycle_mult,
        "WEIGHT_DECAY": args.weight_decay,
    }
    config["WORKER_GRAD_ACCUMULATION_BATCH_SIZE"] = config["TOTAL_BATCH_SIZE"] // config["WORLD_SIZE"]
    print(config)
    assert config["WORLD_SIZE"] > 0, "WORLD_SIZE must be greater than 0 (ensure GPUs are available)."
    assert config["TOTAL_BATCH_SIZE"] % config["WORLD_SIZE"] == 0, "TOTAL_BATCH_SIZE must be divisible by WORLD_SIZE."
    
    
    assert config["WORKER_GRAD_ACCUMULATION_BATCH_SIZE"] > 0, "WORKER_GRAD_ACCUMULATION_BATCH_SIZE must be positive."
    assert config["WORKER_BATCH_SIZE"] <= config["WORKER_GRAD_ACCUMULATION_BATCH_SIZE"], "WORKER_BATCH_SIZE must not exceed WORKER_GRAD_ACCUMULATION_BATCH_SIZE."
    
    mp.spawn(main, args=(config,), nprocs=config["WORLD_SIZE"])