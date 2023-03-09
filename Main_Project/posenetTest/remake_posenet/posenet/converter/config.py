import yaml
import os

BASE_DIR = os.path.dirname(__file__) #현재 파일 경로('./')
config_name='config.yaml' # file name

def load_config():
    cfg_f = open(os.path.join(BASE_DIR, config_name), "r+") #config.yaml 파일 열기
    cfg = yaml.load(cfg_f, Loader = yaml.FullLoader) #yaml 리더로 읽어서 가지고오기
    return cfg
