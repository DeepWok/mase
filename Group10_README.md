This is a test guidline for evaluating the prune methods desgined by group 10.

To run the test file in machop/test/passes/graph/transforms/prune/Group10_test.py, please first train/select a checkpoint to train and load it at original_path in Group_test.py.
Then run the pruning in the command line with:
./ch transform --config PATH_TO_PRUNING_TOML_CONFIG --load PATH_TO_CHECKPOINT --load-type pl --task cls --cpu=0
(config path defalut in /mase/machop/configs/tests/prune/Group10/vgg7_tensor_element.toml, and ALL other config files are located under this path)
and retraining command:
./ch transform --config PATH_TO_RETRAINING_TOML_CONFIG --load PATH_TO_CHECKPOINT --load-type pl --task cls --cpu=0
(config path defalut in /mase/machop/configs/tests/prune/Group10/vgg7_retrain.toml)
Lastly load the pruned state dictionary and retrained model at pruned_path and retrained_path respectively.

-----------------------------------------------------------------------------

Here are some description of majority changes done to the existing model:

1. machop/chop/passes/graph/transforms/pruning/pruning_methods.py  -- Add pruning methods for different scopes, including tensor-element-dimension, tensor-channel-dimension, layer-element-dimension, layer-channel-dimension and global.

2. machop/chop/passes/graph/transforms/pruning/load.py -- Add the corresponding methods to the list.

3. machop/chop/actions/transform.py -- Add output to view sparsity; Add a retrain function so that retrain can be called from a command line; Add some save models for use in the test.

4. machop/chop/passes/graph/transforms/pruning/prune.py The weight pruning in this code (line 28) is to store the mask as a FakeSparseWeight class in the layer (line 153). The mask is not really stored, 
but it can be traced back through the FakeSparseWeight and pruning is done through this class when retraining. Thus when pruning the model, the weights are not really pruned when the pruning pass is called, 
i.e., they are set to 0. So when you want to evaluating the pruning process, you need to load the pruned weights into the existing model now.

5. machop/test/passes/graph/transforms/prune/Group10_test.py Add test code to evaluate the effect of pruning, including testing the model at different times: before pruning, after pruning and after retraining.

6. Some minor bugs on arranging model/weights on different devices (cpu and cuda) were fixed.


Note: The retraining process now support cifar10 and imagnet dataset trained model only, please add more test for other datasets. The test experiments were done with vgg7 and resnet18 trained with cifar10.