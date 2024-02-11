import os
try:
    os.environ['HTTP_PROXY']='http://185.46.212.90:80'
    os.environ['HTTPS_PROXY']='https://185.46.212.90:80'
    print ("proxy_exported")
except:
    None
