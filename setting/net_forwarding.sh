check ip -4 addr show eth0 | grep inet

# run in powershell
netsh interface portproxy add v4tov4 listenport=7860 listenaddress=0.0.0.0 connectport=7860 connectaddress=172.22.245.142

# removing
netsh interface portproxy delete v4tov4 listenport=7860 listenaddress=0.0.0.0
