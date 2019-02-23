import json

file_name = 'data_skippy_cifar10_big_one_test_run_eval_train_hist.json'
eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))

for i in eval_data:
    print(i)
