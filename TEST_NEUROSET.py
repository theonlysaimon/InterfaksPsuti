import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("models/essays")

model = GPT2LMHeadModel.from_pretrained("models/essays")

model.cuda()


text = "<s>Тема: «Создает человека природа, но развивает и образует его общество». (В.Т. Белинский)\nСочинение: "
inpt = tok.encode(text, return_tensors="pt")

out = model.generate(inpt.cuda(), max_length=500, repetition_penalty=5.0, do_sample=True, top_k=5, top_p=0.95, temperature=1)

tok.decode(out[0])
