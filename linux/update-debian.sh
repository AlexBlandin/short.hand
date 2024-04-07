echo "This is meant for Debian Release updates, do not use normally!"
echo "Switch /etc/apt/sources.list from tracking <$1> to tracking <$2>."
read -p "Press <ENTER> to confirm you are willing to do this."

sudo sed -i "s/$1/$2/g" /etc/apt/sources.list

sudo apt update -qy
sudo apt full-upgrade -qy

sudo apt-get update -qy
sudo apt-get upgrade -qy
sudo apt-get dist-upgrade -qy
sudo apt-get check -qy
sudo apt-get autoclean -qy
sudo apt autopurge -qy

cat /etc/os-release
