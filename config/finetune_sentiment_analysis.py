out_dir = 'out-sentiment-analysis-finetune'

always_save_checkpoint = False

wandb_log = True
wandb_project = 'sentiment-analysis'
wandb_run_name = 'mini-gpt-finetune'

dataset = 'customer_service_finetune'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 200

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
sentiment_classifier = True

eval_interval = 100  # decreased to prevent overfitting
eval_iters = 200
log_interval = 50 