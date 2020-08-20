# MetricNet
Online Filtering Training Samples for Robust Visual Tracking (ACM MM2020)

## Results
| OTB2015 | Success | Precision |
|:------- |:----------------:|:----------------:|
| MDNet           | 0.671 | 0.904|
| MDNet+MetricNet | 0.681 | 0.910 |
| ECO           | 0.666 | 0.903|
| ECO+MetricNet | 0.678 | 0.926 |
| ATOM           | 0.665 | 0.870|
| ATOM+MetricNet | 0.675 | 0.881 |

| UAV123 | Success | Precision |
|:------- |:----------------:|:----------------:|
| MDNet           | 0.540 | 0.754|
| MDNet+MetricNet | 0.561 | 0.789 |
| ECO           | 0.533 | 0.764|
| ECO+MetricNet | 0.546 | 0.786 |
| ATOM           | 0.621 | 0.832|
| ATOM+MetricNet | 0.650 | 0.866 |

| LaSOT | Success | Norm Precision |
|:------- |:----------------:|:----------------:|
| MDNet           | 0.390 | 0.430|
| MDNet+MetricNet | 0.443 | 0.523 |
| ECO           | 0.371 | 0.431|
| ECO+MetricNet | 0.419 | 0.501 |
| ATOM           | 0.503 | 0.574|
| ATOM+MetricNet | 0.535 | 0.614 |

## Requirments 
python 3.7  
pytorch  
ubuntu 16.04 + cuda-9.0  

## Installation
The pretrained models are also downloaded.
```bash
bash install.sh conda_install_path metricnet
```  



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

