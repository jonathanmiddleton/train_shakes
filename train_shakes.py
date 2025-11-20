import os, time, math
import numpy as np
import torch
from model import GPTConfig, GPT

#--- Logging ---#
log_interval = 10 # note: this will also trigger host/device sync but you wont notice perf overhead in this case
wandb_log = True # optional W&B logging
wandb_project = 'shakespeare-char'
wandb_run_name = 'overfit '

#--- Model ---#
n_layer = 6
n_head = 6
n_embd = 384
bias = False # do we use bias inside LayerNorm and Linear layers
skip_attn_layers = [] # [2,4] will skip attention in layers 2 and 4 (0-based)
tied_embeddings = True  # whether to use the embedding weights for the output layer as well (GPT-2=True)

#--- Harness ---#
dataset = 'shakespeare_char'
device = 'cuda' if torch.cuda.is_available() else 'mps'
torch_compile = True
max_iters = 5000
gradient_accumulation_steps = 1
dtype='bfloat16' # numeric data type we'll use with autocast, though many ops will end up cast to fp16 on mps
batch_size = 64  # number of independent sequences to process in parallel, gradients averaged across all batches
block_size = 256 # max sequence length
vocab_size = 65 # I think it should be 64 but was 65 when I tested
dropout = 0.2  # N% chance any neuron output is set to zero during training, prevents overfitting

#--- Optimizer ---#
decay_lr = True # whether to decay the learning rate
decay_lr_schedule = 'cosine' # linear or cosine, karpathy used cosine
lr_decay_iters = 5000
max_lr = 1e-3 # with baby networks can afford to go a bit higher
min_lr = 1e-4 # max_lr / 10 usually
s_lr = 1.0  # testing convenience to scale both learning rates
max_lr *= s_lr
min_lr *= s_lr
# betas: roughly 0.9=last 10 steps, 0.99=last 100 steps, ...
beta1 = 0.9  # controls how quickly we react to changes in the gradient based on prior steps
beta2 = 0.99 # controls how much we smooth the gradient based on prior steps
weight_decay = 0  # reduces overfitting by smoothing weights (Karpathy used 1e-1, 0=None)
grad_clip = 1.0 # rescales (smooths) gradients above the normed threshold, or disable if == 0.0, (Karpathy used 1.0)

#--- Setup ---#
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
config = {k: globals()[k] for k in config_keys}
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

tokens_per_iter = gradient_accumulation_steps *  batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337) # globally sets the random seed
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'

data_dir = os.path.join('data', dataset)
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,)) # CPU-tensor
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]) # CPU-tensor
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]) # CPU-tensor    # Optional sanity check (runs on CPU, fails early if something is wrong)
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True) # async transfer to device
    else:
        x, y = x.to(device), y.to(device) # on MPS, storage is already in unified memory; pinning is an error
    return x, y

#--- Model Initialization ---#
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, skip_attn_layers=skip_attn_layers, tied_embeddings=tied_embeddings)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay, max_lr, (beta1, beta2))

if torch_compile:
    print("compiling the model... (takes a ~minute)")
    torch._inductor.config.coordinate_descent_tuning = True if torch.cuda.is_available() else False
    torch._dynamo.config.compiled_autograd = True
    model = torch.compile(model)

def get_lr_linear(it):
    if it < max_iters - lr_decay_iters:
        return max_lr
    lr_scale = (max_iters - it) / lr_decay_iters
    return max_lr * lr_scale

def get_lr_cosine(it):
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = it / lr_decay_iters
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * min_lr

if decay_lr_schedule == 'cosine':
    lr_schedule = get_lr_cosine
elif decay_lr_schedule == 'linear':
    lr_schedule = get_lr_linear
else:
    raise ValueError(f'unknown lr schedule: {decay_lr_schedule}')

#--- Training Loop ---#
iter_num = 0
best_val_loss = 1e9
X, Y = get_batch()
t0 = time.time()
acc_time = 0.0
while True:
    lr = lr_schedule(iter_num) if decay_lr else max_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            # queue on device: forward pass
            logits, loss = model(X, Y)
            # queue on device: scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
        # async host-to-device transfer of next batch during above ops
        X, Y = get_batch()
        loss.backward() # also frees state cached during foward pass
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    acc_time += dt
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        tokens_per_step = batch_size * block_size * gradient_accumulation_steps
        print(f"step {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        if wandb_log: wandb.log({ "step": iter_num, "train/loss": lossf, "lr": lr, "tokens": iter_num * tokens_per_step, "acc_time": acc_time})
        if lossf < 0.1:
            break
    iter_num += 1

    if iter_num > max_iters:
        break
if wandb_log:
    wandb.finish()