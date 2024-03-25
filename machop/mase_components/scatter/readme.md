# Scatter

The scatter module is a fully combinational circuit that takes an unpacked input array of fixed point integers and outputs a low and high precision array each of the same input dimensions of the original input array. The `o_high_precision` array contains the outlier features and zeros in all other positions. The number of outliers conatined within the array is given by the parameter `HIGH_SLOTS`. The remaining features are contained within the `o_low_precision` array with the locations of the outliers repalced with zeros.

The threshold over which a feature is considered an outlier is given by the parameter `THRESHOLD`. The precision of the input values are given by the parameter `PRECISION` and the size of the input array is given by `TENSOR_SIZE_DIM`.

In the case of the input array contaitng more outliers than there are slots of outliers int he `o_high_precision` array, the outliers in the least significant index are given priority. This behviour is replciated in the software model.

## Microarchtiecture

To identify outliers, the absoloute value of each input is compared to the thrrshold value. An array mask of the input array is 