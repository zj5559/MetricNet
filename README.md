# MetricNet
Online Filtering Training Samples for Robust Visual Tracking (ACM MM2020)

## Requirments 
python 3.7
pytorch
ubuntu 16.04 + cuda-9.0

## Installation
#### Install dependencies
```bash
bash install.sh conda_install_path metricnet
```  
#### Download models


## Train
#### Prepare dataset (LaSOT)
```bash
cd Train
python prepare_data.py
```  
#### Train MetricNet
```bash
python train.py
```  
## Eval
#### Integrate MetricNet into MDNet
```bash
cd MDNet_MetricNet
python metric_tracking.py
```  
#### Integrate MetricNet into ECO/ATOM
```bash
cd pytracking_MetricNet/pytracking
python run_tracker.py
```  

