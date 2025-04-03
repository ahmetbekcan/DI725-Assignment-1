# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-sentiment-analysis'
eval_interval = 100  # decreased to prevent overfitting
eval_iters = 200
log_interval = 50 

always_save_checkpoint = False

wandb_log = True
wandb_project = 'sentiment-analysis'
wandb_run_name = 'mini-gpt-train'

dataset = 'customer_service'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 256

sentiment_classifier = True # it will change the model implementation

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4 # decreased since dataset is small
max_iters = 2000
lr_decay_iters = 2000 
min_lr = 1e-5  # learning_rate / 10 usually
beta2 = 0.99 

warmup_iters = 100

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
