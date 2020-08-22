max_lenght = 512
train_batch_size = 8
test_batch_size =  4
epochs =  10
grad_accumulation: 2
model_n = "bert-base-uncased"
training_file = "../data/data.csv"
model_path = "outputs/"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
num_workers = 8
lr: = 1e-5