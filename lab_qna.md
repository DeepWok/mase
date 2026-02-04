# Lab Logbook Answers

## Lab 0

### Tutorial 1

> Task: Delete the call to `replace_all_use_with` to veriyf that FX will report a RuntimeError

Shows error as expected. The runtime error describes:

```
RuntimeError: Tried to erase Node bert_embeddings_dropout but it still had 6 users in the graph:
{getattr_2, size_4, bert_encoder_layer_0_attention_self_query,
 bert_encoder_layer_0_attention_self_key, bert_encoder_layer_0_attention_self_value, add_7}
```

This is because the 6 nodes which were found in the earlier analysis pass still depend on the drouput output, i.e. the nodes have the drouput node in their `args`. FX prevents deletion because it would leave them with invalid inputs. Uncommenting the `replace_all_use_with` and running the analysis pass again shows proper removal, where dropout count is now 0.

### Tutorial 2:

>

### Tutorial 3: QAT


### Tutorial 4: Pruning


### Tutorial 5: NAS Optuna


### Tutorial 6: Mixed Precision Search


### [Hardware Stream] Tutorial ?

