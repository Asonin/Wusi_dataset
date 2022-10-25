# DATA_WUSI

This is a new dataset containing 5 people of high level of interaction. The whole dataset runs under frame rate 25.

Please check our [paper](https://baidu.com) and the [project webpage](https://github.com/Asonin/MRT) for more details. 



## Dependencies

Requirements:
None

## Datasets
We provide the data preprocessing code of [data_wusi](https://github.com/Asonin/MRT).
For data_wusi, the dictionary tree is like
``` 
   data_wusi
   ├── post
   ├── wusi
   |   └──1021_1_
   |   ...
   ├── wusi_nocut
   |   └──1009_2
   |   ...
   ├── data.py
   ├── mix_wusi.py
   ├── process_wusi.py
   ├── README.md
```

## process_wusi.py
This script reads the raw data and cut them up into sequences, you can specify the sequence length by argument _sequence_len_ and stride by the argument _stride_. 
After processing the raw data, it shall tell you the sequence number and sequence length. Then, processed data will be divided into training and testing set. You could specify the ratio of training dataset by argument _ratio_, the remaining should automatically become test set.

### Enter folder data_wusi/, please then try
```
python precess_wusi.py --stride=[your stride] --sequence_len=[your sequence len] --ratio=[your ratio]
```

