max_lenght = 512
train_batch_size = 8
test_batch_size = 4
epochs = 10
models = {"bert": "bert-base-cased", "roberta": "roberta-base"}
grad_accumulation: 2
model = models["roberta"]
training_file = "../data/data.csv"
model_path = "outputs/"
num_workers = 8
lr = 1e-5
data_path = "/GW/Health-Corpus/work/UMLS/data/data.csv"
m =0.3
checkpoint_interval = 10000