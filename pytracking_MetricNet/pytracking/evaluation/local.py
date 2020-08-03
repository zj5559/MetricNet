from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = '/media/zj/4T/Dataset/LaSOT/dataset/images'
    settings.network_path = '/home/zj/Downloads/pytracking-master/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/media/zj/4T/Dataset/OTB-100/'
    settings.results_path = '/home/zj/tracking/metricNet/MetricNet-git/pytracking_MetricNet/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/zj/4T/Dataset/TrackingNet'
    settings.uav_path = '/media/zj/4T/Dataset/UAV123/Dataset_UAV123/UAV123'
    settings.vot_path = '/media/zj/4T/Dataset/VOT/VOT18'

    return settings

