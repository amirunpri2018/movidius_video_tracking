##
## Inference Mode
## 

import requests
import yaml
import logging

config = yaml.load(open("./CONFIG", "rb"))
logger = logging.getLogger(__name__)

def write_api(tracks):
    # insert device_id 
    for track in tracks:
        track['device_id'] = config.get("device_id", "0")
    payload = {"tracks": tracks}

    # api url 
    api_config = config.get('api', {})
    host_url = api_config.get('host_url', '127.0.0.1:5000')
    tracks_endpoint = api_config.get('tracks_endpoint', '/tracks')
    url = host_url + tracks_endpoint
    resp = requests.post(url, json=payload)
    
    logger.info("post to %s" % url)
    logger.info("[%s] %s" % (resp.status, resp.json()))
