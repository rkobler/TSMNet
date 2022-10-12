#!/bin/bash

### metntal imagery
## inter session TL
datasets=bnci2014001,bnci2015001,lee2019,stieger2021
# comparision to baseline methods
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=eegnet,shconvnet
# ablation study
[ $? -eq 0 ] && datasets=bnci2014001,bnci2015001,lee2019,stieger2021
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_sppddsbn,cnnnet_dsmbn
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=tsmnet_sppddsbn,cnnnet_dsmbn

## inter subject TL
[ $? -eq 0 ] && datasets=bnci2014001,bnci2015001,lee2019,stieger2021_last
# comparision to baseline methods
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject dataset=$datasets nnet=eegnet,shconvnet
# ablation study
[ $? -eq 0 ] && datasets=bnci2014001,bnci2015001,lee2019,stieger2021_last
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject dataset=$datasets nnet=eegnet,shconvnet

### mental workload estimation
## inter session TL
datasets=hinss2021
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=eegnet,shconvnet
## inter subject TL
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject dataset=$datasets nnet=eegnet,shconvnet