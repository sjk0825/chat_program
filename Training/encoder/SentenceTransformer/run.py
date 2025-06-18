
import json
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator # cosine, dot, uclidean

config = './config/common.json'
with open(config,'r',encoding='utf-8') as f:
    config = json.load(f)

model = SentenceTransformer(
    config['SentenceTransformer']['name'],
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on AllNLI triplets",
    )
)

dataset = load_dataset(config['dataset']['name'], "triplet")
train_dataset = dataset["train"].select(range(config['dataset']['train_size']))
eval_dataset = dataset["dev"]
test_dataset = dataset["test"]

if config['loss'] == 'MultipleNegativesRankingLoss':
    loss = MultipleNegativesRankingLoss(model)
else:
    raise NotImplementedError 'add loss logic'

training_args = config['SentenceTransformerTrainingArguments']
args = SentenceTransformerTrainingArguments(
    output_dir=training_args['output_dir'],
    num_train_epochs=training_args[''],
    per_device_train_batch_size=training_args[''],
    per_device_eval_batch_size=training_args[''],
    learning_rate=training_args['learning_rate'],
    warmup_ratio=training_args['warmup_ratio'],
    fp16=training_args['fp16'], 
    bf16=training_args['bf16'], 
    batch_sampler=BatchSamplers.NO_DUPLICATES, 
    eval_strategy=training_args['eval_strategy'],
    eval_steps=training_args['eval_steps'],
    save_strategy=training_args['save_strategy'],
    save_steps=training_args['save_steps'],
    save_total_limit=training_args['save_total_limit'],
    logging_steps=training_args['logging_steps'],
    run_name=training_args['run_name'],  
)

if evaluator == 'TripletEvaluator':
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="all-nli-test",
    )
else:
    raise NotImplementedError "please implement evaluator logic"
dev_evaluator(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    loss=loss,
    evaluator=dev_evaluator, 
)
trainer.train()

test_evaluator(model)

model.save_pretrained(config['save'])

model.push_to_hub(config['push_to_hub'])