checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

from pathlib import Path
from chop import MaseGraph
from chop.tools import get_tokenized_dataset, get_trainer
import chop.passes as passes
import json

people = ["amin", "ali", "harun"] # idress.json is half done before cancellation
sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
for person in people: 
    eval_dict = {}
    eval_dict["l1norm"] = {"no finetuning": {}, "with finetuning": {}}
    eval_dict["Random"] = {"no finetuning": {}, "with finetuning": {}}
    # l1norm method
    for sparsity in sparsities:
        # Load starting model
        mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_3_qat")
        # Load dataset, tokenizer
        dataset, tokenizer = get_tokenized_dataset(
            dataset=dataset_name,
            checkpoint=tokenizer_checkpoint,
            return_tokenizer=True,
        )
        
        # Unstructured Pruning
        pruning_config = {
            "weight": {
                "sparsity": 0.5,
                "method": "l1-norm",
                "scope": "local",
            },
            "activation": {
                "sparsity": 0.5,
                "method": "l1-norm",
                "scope": "local",
            },
        }
        pruning_config["weight"]["sparsity"] = sparsity
        pruning_config["activation"]["sparsity"] = sparsity
        
        mg, _ = passes.prune_transform_pass(
            mg,
            pass_args=pruning_config,
        )

        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=5,
        )

        # Evaluate accuracy
        eval_results = trainer.evaluate()
        print(f"Unstructured Pruning (no finetuneing): Evaluation accuracy for sparsity {sparsity}: {eval_results['eval_accuracy']}")
        eval_dict["l1norm"]["no finetuning"][sparsity] = eval_results['eval_accuracy']

        # Finetune
        trainer.train()
        
        eval_results = trainer.evaluate()
        print(f"Unstructured Pruning (finetuning): Evaluation accuracy for sparsity {sparsity}: {eval_results['eval_accuracy']}")
        
        # Save eval results
        eval_dict["l1norm"]["with finetuning"][sparsity] = eval_results['eval_accuracy']

        # continually dump to json
        json.dump(eval_dict, open(f"task_4_eval_dict_{person}.json", "w"))
        
    # Random method
    for sparsity in sparsities: 
        # Load starting model
        mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_3_qat")
        # Load dataset, tokenizer
        dataset, tokenizer = get_tokenized_dataset(
            dataset=dataset_name,
            checkpoint=tokenizer_checkpoint,
            return_tokenizer=True,
        )
        
        # Unstructured Pruning
        pruning_config = {
            "weight": {
                "sparsity": 0.5,
                "method": "random",
                "scope": "local",
            },
            "activation": {
                "sparsity": 0.5,
                "method": "random",
                "scope": "local",
            },
        }
        pruning_config["weight"]["sparsity"] = sparsity
        pruning_config["activation"]["sparsity"] = sparsity
        
        mg, _ = passes.prune_transform_pass(
            mg,
            pass_args=pruning_config,
        )

        trainer = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=5,
        )

        # Evaluate accuracy
        eval_results = trainer.evaluate()
        print(f"Unstructured Pruning (no finetuneing): Evaluation accuracy for sparsity {sparsity}: {eval_results['eval_accuracy']}")
        eval_dict["Random"]["no finetuning"][sparsity] = eval_results['eval_accuracy']

        # Finetune
        trainer.train()
        
        eval_results = trainer.evaluate()
        print(f"Unstructured Pruning (finetuning): Evaluation accuracy for sparsity {sparsity}: {eval_results['eval_accuracy']}")
        
        # Save eval results
        eval_dict["Random"]["with finetuning"][sparsity] = eval_results['eval_accuracy']
        # continually dump to json
        json.dump(eval_dict, open(f"task_4_eval_dict_{person}.json", "w"))