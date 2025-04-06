# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-sentiment-analysis-finetune'

always_save_checkpoint = False

wandb_log = True
wandb_project = 'sentiment-analysis'
wandb_run_name = 'mini-gpt-finetune'

dataset = 'customer_service'
init_from = 'gpt2'

batch_size = 1
gradient_accumulation_steps = 32

learning_rate = 3e-5
decay_lr = False

sentiment_classifier = True # it will change the model implementation

# #changes
eval_interval = 100  # decreased to prevent overfitting
eval_iters = 200
log_interval = 50 
max_iters = 2000
warmup_iters = 100