import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
print(''.join(chars))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

import torch
import torch.nn as nn
from torch.nn import functional as F

data = torch.tensor(encode(text),dtype=torch.long)
print(data.shape,data.dtype)

#split into train and test data set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd =384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split =="train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb,yb = x.to(device),y.to(device)
    return xb, yb
xb, yb = get_batch('train')
print(xb)

# bag-of-words toy demo: illustrates the causal averaging trick behind self-attention
torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
x.shape
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        # sum up and mean all values before x, not after
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev,0)

#more efficient, use matrix multp
wei = torch.tril(torch.ones(T,T))
wei = wei/wei.sum(1,keepdim=True)
wei
xbow2 = wei @ x
# summing up per batch per channel avg of all token
# values of itself and all values before it
xbow2

# use softmax
wei = torch.zeros((T,T))
tril = torch.tril(torch.ones((T,T)))
wei = wei.masked_fill(tril==0,float('-inf'))
wei = F.softmax(wei,dim=-1)
xbow3 = wei@x
torch.allclose(xbow,xbow3)

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        # weight matrix of the key v, and q,
        # linear chains operations together through matmult
        self.key = nn.Linear(n_embd,head_size,bias=False)
        # n embed is the actual dim size of input token x, so its input size,
        # then head size is the dim of how u want to represent the input in the model, usually smnaller
        # (with multi-head attention, n_embd is split across heads: head_size = n_embd / n_head)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        # es a tensor as part of the module's state,
        # without treating it as a learnable parameter — it's PyTorch's way of saying "this data belongs to the model and should move around with it, but training should never update it."
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        #batch size, time steps, channels
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q@k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei@v
        return out

class MultiHeadAttention(nn.Module):

    """running multiple heads in parallel"""

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        # combining outputs together from all heads and concat them
        self.proj = nn.Linear(n_embd,n_embd)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out

#think of what they found from other tokens
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
            # randomly prevent the nodes from communicating
            # randomly set some neurons output to 0, train without them
            # prevent overfitting overreliance on certain neurons
            # training an ensembe of subnetworks and merge them
            # a regularisation technique
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # one row per character, look up js grab that row
        # matrix that stores every embedding
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        # divide n_embd by num_heads so that when the heads' outputs are concatenated,
        # the total size comes back to n_embd (what the rest of the model expects)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)],nn.LayerNorm(n_embd))
        # output predictions for vocab size num and embed is the num of embeding for prev token
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape

        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T,device=device))
        x = token_embed+pos_embed
        # apply self attention to embeddings
        x = self.blocks(x)
        # project token representation to vocab size number s which are prob
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            #flatten dim as cross entrop expects 2 dim
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so position embedding never runs out of rows
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = m(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

m = BigramLanguageModel(vocab_size)
m.to(device)

optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)

print(device)
for steps in range(max_iters):
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))


"""
This is decoder only, unlike in intital paper which is encoder decoder
as it was for machine translation, encode the french text,
decode english text.
Here it is only generating bnased on past texts, masked self attention,
so js use decoder
If use encoder, all texts in encoder output can talk to the decoder ouitputs in attention at all steps
"""