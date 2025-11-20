### Changes
- super simplified for shakespeare_char on mps/cuda
- train.bin has 500k chars
- no eval
- added optional linear learning rate decay
- kept most of karpthy's comments and added additional clarifying comments
- sanity check compile on/off - **off** may be faster for you 
- no config file, it's at the top of train_shakes.py

### Packages
- packages pinned in the original requirements.txt are very outdated
- upgrade all packages in this requirements.txt 
- you can try torch 2.10.0-pre but 2.9.0 is probably fine

### Notes
- to save time you can try to get to loss of 2 by around 400 steps before trying to get below 0.1
- it's possible to get loss < 0.1 before 3,300 steps - probably well before
- it's worth setting up (free) [W&B logging](https://wandb.ai) so you have history and graphing 