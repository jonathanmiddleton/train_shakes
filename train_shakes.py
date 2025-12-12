import os, time, math, sys, itertools, ast
import numpy as np
import torch
from model import GPTConfig, GPT
import wandb as _wandb

#--- Logging ---#
log_interval = 1 # note: this will also trigger host/device sync but you wont notice perf overhead in this case
wandb_log = True # optional W&B logging
wandb_project = 'shakespeare-char'
wandb_group = 'debug2'
wandb_run_name = 'config'

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
torch_compile = False   # NOTE: source of randomness which affects reproducibility
max_iters = 3500
gradient_accumulation_steps = 1
dtype='bfloat16' # numeric data type we'll use with autocast, though many ops will end up cast to fp16 on mps
batch_size = 64  # number of independent sequences to process in parallel, gradients averaged across all batches
block_size = 256 # max sequence length
vocab_size = 65 # I think it should be 64 but was 65 when I tested
dropout = 0.0  # N% chance any neuron output is set to zero during training, prevents overfitting

#--- Optimizer ---#
decay_lr = True # whether to decay the learning rate
decay_lr_schedule = 'linear' # linear or cosine, karpathy used cosine
lr_decay_iters = 1500
max_lr = 1e-3 # with baby networks can afford to go a bit higher
min_lr = 1e-4 # max_lr / 10 usually
s_lr = 1.1  # testing convenience to scale both learning rates (applied per-run in run_one)
# betas: roughly 0.9=last 10 steps, 0.99=last 100 steps, ...
beta1 = 0.8  # controls how quickly we react to changes in the gradient based on prior steps
beta2 = 0.999 # controls how much we smooth the gradient based on prior steps
weight_decay = 0  # reduces overfitting by smoothing weights (Karpathy used 1e-1, 0=None)
grad_clip = 0.0 # rescales (smooths) gradients above the normed threshold, or disable if == 0.0, (Karpathy used 1.0)

def make_default_config():
    # Snapshot of simple-typed globals as defaults
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
    return {k: globals()[k] for k in config_keys}


def parse_value(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        # allow plain strings without quotes
        return s


def run_one(config: dict, override_name_suffix: str = ""):
    # Unpack frequently used config values with locals
    log_interval = config['log_interval']
    wandb_log = config['wandb_log']
    wandb_project = config['wandb_project']
    wandb_group = config['wandb_group']
    wandb_run_name = config['wandb_run_name'] + (override_name_suffix if override_name_suffix else "")

    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    bias = config['bias']
    skip_attn_layers = config['skip_attn_layers']
    tied_embeddings = config['tied_embeddings']

    dataset = config['dataset']
    device = config['device']
    torch_compile = config['torch_compile']
    max_iters = config['max_iters']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    # dtype is kept for compatibility (currently uses bfloat16 below)
    batch_size = config['batch_size']
    block_size = config['block_size']
    vocab_size = config['vocab_size']
    dropout = config['dropout']

    decay_lr = config['decay_lr']
    decay_lr_schedule = config['decay_lr_schedule']
    lr_decay_iters = config['lr_decay_iters']
    max_lr = config['max_lr']
    min_lr = config['min_lr']
    s_lr = config['s_lr']
    beta1 = config['beta1']
    beta2 = config['beta2']
    weight_decay = config['weight_decay']
    grad_clip = config['grad_clip']

    # Apply learning rate scaling per run so grid overrides like s_lr=... take effect at runtime
    base_max_lr = max_lr
    base_min_lr = min_lr
    max_lr = base_max_lr * s_lr
    min_lr = base_min_lr * s_lr

    # --- Setup per run ---
    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    torch.manual_seed(1337)
    device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'

    data_dir = os.path.join('data', dataset)
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))  # CPU-tensor
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])  # CPU-tensor
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])  # CPU-tensor
        if device_type == 'cuda':
            x_dev = x.pin_memory().to(device, non_blocking=True)
            y_dev = y.pin_memory().to(device, non_blocking=True)
        else:
            x_dev = x.to(device)
            y_dev = y.to(device)
        return x_dev, y_dev

    # --- W&B per run ---
    if wandb_log:
        _wandb.init(project=wandb_project, group=wandb_group, name=wandb_run_name, config=config)

    # --- Model Initialization ---
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout,
        skip_attn_layers=skip_attn_layers,
        tied_embeddings=tied_embeddings,
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(weight_decay, max_lr, (beta1, beta2))

    if torch_compile:
        print("compiling the model... (takes a ~minute)")
        # torch._inductor.config.coordinate_descent_tuning = True if torch.cuda.is_available() else False
        # torch._dynamo.config.compiled_autograd = True
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
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * min_lr

    if decay_lr_schedule == 'cosine':
        lr_schedule = get_lr_cosine
    elif decay_lr_schedule == 'linear':
        lr_schedule = get_lr_linear
    else:
        raise ValueError(f'unknown lr schedule: {decay_lr_schedule}')

    # --- Training Loop per run ---
    iter_num = 0
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
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch()  # prefetch next batch during compute
            loss.backward()

        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        acc_time += dt
        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            tokens_per_step = batch_size * block_size * gradient_accumulation_steps
            print(f"step {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
            if wandb_log:
                _wandb.log({
                    "step": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "tokens": iter_num * tokens_per_step,
                    "acc_time": acc_time,
                    "final_step": iter_num if lossf < 0.1 else None
                })
            if lossf < 0.1:
                break
        iter_num += 1

        if iter_num > max_iters:
            break

    if wandb_log:
        _wandb.finish()
    # try to free GPU memory between runs
    try:
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv):
    default_config = make_default_config()

    # Collect overrides of the form key=v1,v2
    overrides = {}
    for token in argv:
        if '=' not in token:
            raise ValueError(f"Override must be key=val[,val2,...], got: {token}")
        key, raw_vals = token.split('=', 1)
        key = key.strip()
        if key not in default_config:
            raise KeyError(f"Unknown config key: {key}")
        values = [parse_value(v) for v in raw_vals.split(',')]
        overrides[key] = values

    if not overrides:
        # No grid search, run once with defaults
        run_one(default_config.copy(), override_name_suffix="")
        return

    # Build Cartesian product of overrides
    keys = list(overrides.keys())
    grid_values = [overrides[k] for k in keys]
    for combo in itertools.product(*grid_values):
        # Prepare config for this run
        cfg = default_config.copy()
        suffix_parts = []
        for k, v in zip(keys, combo):
            cfg[k] = v
            suffix_parts.append(f"{k}={v}")
        suffix = " " + ",".join(suffix_parts)
        run_one(cfg, override_name_suffix=suffix)


if __name__ == '__main__':
    main(sys.argv[1:])