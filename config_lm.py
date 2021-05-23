max_length = 256
train_batch_size = 8
test_batch_size = 4
epochs = 10
models = {
    "bert": "bert-base-cased",
    "roberta": "roberta-base",
    "albert-base": "albert-base-v2",
    "albert-large": "albert-large-v2",
}
grad_accumulation: 2
model = models["albert-large"]
training_file = "../data/data.csv"
model_path = "outputs/"
num_workers = 8
lr = 1e-5
data_path = "/GW/Health-Corpus/work/UMLS/data/data.csv"
m = 0.3
checkpoint_interval = 10000
optimizer = "adam"
load = False
weight_decay = 0.0005
