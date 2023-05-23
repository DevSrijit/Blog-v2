---
title: 'Can I become the next Shakespeare ?'
description: 'Experimenting with GPT-2.'
date: 2023-05-23
tags:
  - posts
layout: layouts/post.njk
---

Have you ever wondered what it would be like to write like Shakespeare? To be able to craft beautiful sonnets and plays that have stood the test of time? Well, I certainly have. And that's why I decided to embark on a journey to see if I could become the next Shakespeare, with the help of GPT-3.

### The Inspiration

I was inspired by a video I saw on YouTube, where a programmer used GPT-2 to generate text that sounded like Shakespeare. The results were impressive, and I was intrigued. Could I do the same thing? Could I use GPT-3 to generate text that sounded like Shakespeare?

### The Setup

To get started, I needed to set up my environment. I decided to use Google Colab with a T4 GPU to perform this task. I also needed to install PyTorch and import some libraries.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from google.colab import drive
drive.mount(‘/content/drive’)
```

### The Data

Next, I needed some data to train my model on. I decided to use the works of Shakespeare, which I downloaded from [here](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt). I then created a mapping from characters to integers, which would be used to encode the text.

```python
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
```

### The Model

Now it was time to create the model. I decided to use a bigram language model, which would predict the next character given the current character. The model consisted of a token embedding table, a position embedding table, several transformer blocks, a layer normalization layer, and a linear layer.

```python
class BigramLanguageModel(nn.Module):

def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss
```

### The Training

With the model in place, it was time to train it. I used the AdamW optimizer and cross-entropy loss. I also evaluated the loss on the train and validation sets every 100 iterations.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### The Results
After training the model for 100,000 iterations, I generated some text using the model. The results were impressive, with the model generating text that sounded like Shakespeare.

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```

Here is a sample of the generated text:

```
JULIET:
When yet shed did forward, the dish weather bequar and father
but and build into me; but satisfied
not imputeous office, steel;
I could say it prayer, better? my son,
Is Romeo: the stession
That he reak offence,--more? than we call the other world
And to this sanctuary fellow: if Godoen Richard,
Sir, not it mine Escaptaine: engs dood.

Second Servingman:
Be rid your father on, sir; if your kind for Fortune to my bound do before thee,
Either now, to't! Ply Tybaln!
Besides! whither, good kinsman, sirrah,
Whither to both lask, wife with thee,
He would sit be the oin; he make
I will all great woon with them.

VIRGILLIO:
Nay, my lord is the hungeral with her?
Whose'er to him. Come, go yie much
Is King Ricquirs bosom, on live;
And so how we then?

Can May derty, and looks so, or much
pater ease: then a lambury
Of the death embrother, and now
If it not do? the hay may had lew take the housand, that bear me said just,
And he showly herefore: Romeo let hear and Romeo! What but ye we would I cannot comfort wife;
We may so be safe-her. We have not very
counsels are petty of the counsel-graved feel at this,
With hell kill themself must be roguing Edward's oaths.

DUCHESSS OF YORK:
Away; and scain of the bosow, this of his second mewift out!
O, how if such every thing, by my pating soul's death.
Yo marry his deed in
to yield believe, Grey? Paria,
Not's of friends to Hontesterty like
To out of Him goise.
```

### Conclusion

In conclusion, I was able to use GPT-3 to generate text that sounded like Shakespeare. While I may not be the next Shakespeare, this project was a fun and interesting way to explore the capabilities of language models and their potential applications. With further training and fine-tuning, it’s possible that these models could be used to generate more complex and nuanced text, such as poetry or even entire novels. The possibilities are endless, and I’m excited to see where this technology will take us in the future. 
I hope you find this helpful!