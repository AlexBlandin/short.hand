
sudo sed -i 's/previous/newdebian/g' /etc/apt/sources.list

sudo apt update -qy
sudo apt full-upgrade -qy

sudo apt-get update -qy
sudo apt-get upgrade -qy
sudo apt-get dist-upgrade -qy
sudo apt-get check -qy
sudo apt-get autoclean -qy
sudo apt --purge autoremove -qy

cat /etc/os-release
