### this is for running in local ###
import os
try:
    os.environ['HTTP_PROXY']='http://185.46.212.90:80'
    os.environ['HTTPS_PROXY']='https://185.46.212.90:80'
    print ("proxy_exported")
except:
    None

# !apt install foremost

# +
#export HTTP_PROXY="http://185.46.212.90:80"
#export HTTPS_PROXY="https://185.46.212.90:80"
