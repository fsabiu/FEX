sudo apt -y install firewalld
sudo firewall-cmd --zone=public --add-port=2053/tcp --permanent
sudo firewall-cmd --zone=public --add-port=2054/tcp --permanent
sudo firewall-cmd --zone=public --add-port=2055/tcp --permanent
sudo firewall-cmd --reload