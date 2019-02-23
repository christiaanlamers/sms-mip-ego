import json

file_name = 'skippy_test_train_hist_eval_train_hist.json'
eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))

for i in eval_data:
    print(i)
