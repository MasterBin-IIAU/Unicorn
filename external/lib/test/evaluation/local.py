from lib.test.evaluation.environment import EnvSettings
from unicorn.data import get_unicorn_datadir
import os

def local_env_settings():
    settings = EnvSettings()
    # Set your local paths here.
    settings.prj_dir = os.path.join(get_unicorn_datadir(), "..")
    settings.data_dir = get_unicorn_datadir()
    settings.got10k_path = os.path.join(settings.data_dir, 'GOT10K')
    settings.lasot_path = os.path.join(settings.data_dir, 'LaSOT')
    settings.trackingnet_path = os.path.join(settings.data_dir, 'TrackingNet')
    settings.nfs_path = os.path.join(settings.data_dir, 'nfs')
    settings.otb_path = os.path.join(settings.data_dir, 'OTB2015')
    settings.uav_path = os.path.join(settings.data_dir, 'UAV123')
    settings.vot_path = os.path.join(settings.data_dir, 'VOT2019')
    settings.result_plot_path = os.path.join(settings.prj_dir, 'test/result_plots')
    settings.results_path = os.path.join(settings.prj_dir, 'test/tracking_results')    # Where to store tracking results
    settings.segmentation_path = os.path.join(settings.prj_dir, 'test/segmentation_results')

    return settings

