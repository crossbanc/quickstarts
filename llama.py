# -*- coding: utf-8 -*-

# -- Sheet --

import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as f
from collections import OrderedDict 

# Simplified model parameters 
params = {     
    
    'context_window': 16,
    'd_model': 128,          
    'log_interval': 10,      
    'batch_size': 32, 
    'n_heads': 8,
    'n_layers': 4,
    'epochs': 10000
}

learnenglish = "your_learn_english.txt"
lines = open(learnenglish, 'r').read()
# Create a sorted list of unique characters in the dataset
vocab = sorted(list(set(lines)))

# Mapping integers to characters (itos)
itos = {i: ch for i, ch in enumerate(vocab)}
# Mapping characters to integers (stoi)
stoi = {ch: i for i, ch in enumerate(vocab)}

# Encode function: Converts a string to a list of integers using the mapping stoi
def encode(s):
    return [stoi[ch] for ch in s]

# Decode function: Converts a list of integers back to a string using the mapping itos
def decode(l):
    return ''.join([itos[i] for i in l])

# Alternatively, use the sentencepiece python wrapper but you will need to modify get_batches function. 
# https://github.com/google/sentencepiece/blob/master/python/README.md

# In machine learning or deep learning projects, such train-test splits are crucial for developing and evaluating models, 
# and the same principle applies here. Depending on the use-case, the approach may be different.

dataset = torch.tensor(encode(lines), dtype=torch.int8)

# Function to get batches for training, validation, or testing 
def get_batches(data, split, batch_size, context_window):
    # Split the dataset into training, validation, and test sets
    trainset = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    testset = data[int(.9 * len(data)):]

    # Determine which split to use
    batch_data = trainset
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = testset

    # Pick random starting points within the data
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input sequences (x) and corresponding target sequences (y)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y

@torch.no_grad()  
def evaluate_loss(model, config):

    out = {}
    model.eval()

    # Iterate through training and validation splits
    for split in ["train", "val"]:

        losses = []
        # Generate 10 batches for evaluation
        for _ in range(10):
            # Get input sequences (xb) and target sequences (yb)
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            
            # Perform model inference and calculate the loss
            _, loss = model(xb, yb)
            
            # Append the loss to the list
            losses.append(loss.item())

        # Calculate the mean loss for the split and store it in the output dictionary
        out[split] = np.mean(losses)
    
    # Set the model back to training mode
    model.train()
    
    return out

# Function to perform training
def train(model, optimizer, config, scheduler=None, logs=False):
    # Placeholder for storing losses
    losses = []
    
    # Start tracking time
    start_time = time.time()

    # Iterate through epochs
    for epoch in range(config['epochs']):
        # Zero out gradients
        optimizer.zero_grad()

        # Obtain batches for training
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # Forward pass through the model to calculate logits and loss
        logits, loss = model(xs, targets=ys)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # If a learning rate scheduler is provided, adjust the learning rate
        if scheduler:
            scheduler.step()

        # Log progress every specified interval
        if epoch % config['log_interval'] == 0:
            # Calculate batch time
            batch_time = time.time() - start_time
            
            # Evaluate loss on validation set
            x = evaluate_loss(model)
            
            # Store the validation loss
            losses += [x]
            
            # Print progress logs if specified
            if logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in Secs {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
                
            # Reset the timer
            start_time = time.time()

            # Print learning rate if a scheduler is provided
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # Print the final validation loss
    print("Validation loss: ", losses[-1]['val'])
    
    # Plot the training and validation loss curves
    return pd.DataFrame(losses).plot()



# LLaMA introduces three architectural modifications to the original Transformer

class RMSNorm(nn.Module):

    def __init__(self, layer_shape):
        super(RMSNorm, self).__init__()

        # Registering a learnable parameter 'scale' as a parameter of the module
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """
        Assumes shape is (batch, seq_len, d_model)
        """
        # Calculating the Frobenius norm, RMS = 1/sqrt(N) * Frobenius norm
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5

        # Normalizing the input tensor 'x' with respect to RMS
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)

        # Scaling the normalized tensor using the learnable parameter 'scale'
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

def get_rotary_matrix(context_window, embedding_dim):
    # Initialize a tensor for the rotary matrix with zeros
    r = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)    
    # Loop through each position in the context window
    for position in range(context_window):
        # Loop through each dimension in the embedding
        for i in range(embedding_dim // 2):
            # Calculate the rotation angle (theta) based on the position and embedding dimension
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            # Calculate the rotated matrix elements using sine and cosine functions
            m_theta = position * theta
            r[position, 2 * i, 2 * i] = np.cos(m_theta)
            r[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            r[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            r[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)

    return r

class SwiGLU(nn.Module):
    
    def __init__(self, size):
        super().__init__()
        self.config = params  # Configuration information
        self.linear_gate = nn.Linear(size, size)  # Linear transformation for the gating mechanism
        self.linear = nn.Linear(size, size)  # Linear transformation for the main branch
        self.beta = torch.randn(1, requires_grad=True)  # Random initialization of the beta parameter

        # Using nn.Parameter for beta to ensure it's recognized as a learnable parameter
        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # Swish-Gated Linear Unit computation
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)  # Element-wise multiplication of the gate and main branch
        return out



# Create The Model

class RopeAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Linear transformation for query
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # Linear transformation for key
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # Linear transformation for value
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # Obtain rotary matrix for positional embeddings
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def forward(self, x, return_attn_weights=False):
        # x: input tensor of shape (batch, sequence length, dimension)
        b, m, d = x.shape  # batch size, sequence length, dimension

        # Linear transformations for Q, K, and V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Rotate Q and K using the RoPE matrix
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # Perform scaled dot-product attention
        activations = f.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )

        if return_attn_weights:
            # Create a causal attention mask
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            # Calculate attention weights and add causal mask
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = f.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations

class RopeMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create a list of RopeAttentionHead instances as attention heads
        self.heads = nn.ModuleList([
            RopeAttentionHead(config) for _ in range(config['n_heads'])
        ])
        # Linear transformation after concatenating heads
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])  
        # Dropout layer
        self.dropout = nn.Dropout(.1)  

    def forward(self, x):
        # x: input tensor of shape (batch, sequence length, dimension)

        # Process each attention head and concatenate the results
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        
        # Apply linear transformation to the concatenated output
        x = self.linear(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x

# Add RMSNorm and residual connection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # RMSNorm layer
        self.rms = RMSNorm((config['context_window'], config['d_model']))

        # RoPE Masked Multihead Attention layer
        self.attention = RopeMaskedMultiheadAttention(config)

        # Feedforward layer with SwiGLU activation
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        # one block of attention
        x = self.rms(x) # RMS pre-normalization
        x = x + self.attention(x)  # residual connection

        x = self.rms(x) # RMS pre-normalization
        x = x + self.feedforward(x)  # residual connection
        return x

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding layer for token representations
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        # Sequential block of LlamaBlocks based on the specified number of layers
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        # Feedforward network (FFN) for final output
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # Print total number of parameters in the model
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # Input token indices are passed through the embedding layer
        x = self.embeddings(idx)
        # Process the input through the LlamaBlocks
        x = self.llama_blocks(x)
        # Pass the processed input through the final FFN for output logits
        logits = self.ffn(x)

        # If targets are not provided, return only the logits
        if targets is None:
            return logits
        # If targets are provided, compute and return the cross-entropy loss
        else:
            loss = f.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss



def main():

    #Train and Save the model.
    llama = Llama(params)
    adam_optimizer = torch.optim.Adam(llama.parameters())
    train(llama, adam_optimizer, params)
    torch.save(llama, 'llama_model.pth')

    #See Test Performance.
    txs, tys = get_batches(dataset, 'test', params['batch_size'], params['context_window'])
    tlogits, tloss = llama(txs, tys)
    print(tlogits)
    print(tloss)


    # Create Llama model with Cosine Annealing learning schedule
    llama_with_cosine = Llama(params)

    # Define Adam optimizer with specific hyperparameters
    llama_optimizer = torch.optim.Adam(
        llama.parameters(),
        betas=(.9, .95),
        weight_decay=.1,
        eps=1e-9,
        lr=1e-3
    )

    # Define Cosine Annealing learning rate scheduler
    lrscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)
    # Train the Llama model with the specified optimizer and scheduler
    train(llama_with_cosine, llama_optimizer, params, lrscheduler)
    torch.save(llama, 'llama_cos_model.pth')

    #See Test Performance.
    tlogits, tloss = llama_with_cosine(txs, tys)
    print(tlogits)
    print(tloss)

    return None

# Load the model and update so as to improve the number of paarmeters, fine tune it for your use case or
# https://arxiv.org/abs/2005.11401

# After the model learns english and depending on the use case, modifications to or another get_batch function as discussed.    

if __name__ == '__main__':
    main()



