# ShapeShifter Networks for Neural Parameter Allocation Search

**SSN** contains a pytorch implementation for our [paper](https://arxiv.org/pdf/2006.10598.pdf).  If you find this code useful in your research, please consider citing:

    @InProceedings{plummerNPAS2022,
         author={Bryan A. Plummer and Nikoli Dryden and Julius Frost and Torsten Hoefler and Kate Saenko},
         title={Neural Parameter Allocation Search},
         booktitle={International Conference on Learning Representations (ICLR)},
         year={2022}
    }

This code was tested using pytorch v1.9.


### Training New Models
You can train a model using:

    ./train_models.sh <NUM GPUS> <DATASET> <SHARE TYPE> <NAME OF EXPERIMENT> <ADDITIONAL ARGUMENTS>
    ./train_models.sh 1 cifar100 wavg test_wrn_28_10

You can see a listing and description of many parameter settings with:

    python main.py --help
    
Some key arguments would be:

| arguments  | description |
| ------------- | ------------- |
| --max_params  | indicates the maximum number of parameters used by a model  |
| --param_groups  | number of parameter groups to train  |
| --upsample_type  | parameter upsampling strategy to use  |
| --group_share_type  | parameter downsampling approach to use when learning parameter groups  |

### Evaluation
You can test a model using:

    ./test_models.sh <NUM GPUS> <DATASET> <SHARE TYPE> <NAME OF EXPERIMENT> <ADDITIONAL ARGUMENTS>
    ./test_models.sh 1 cifar100 wavg test_wrn_28_10
