#!/usr/bin/env bash

echo "updating ubuntu ..."
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install htop

# python 3.x extensions
echo "installing Python 3.x extensions ..."
sudo apt-get -y install gcc binutils
sudo apt-get -y install software-properties-common
sudo apt-get -y install python3.10 python3.10-venv python3.10-dev python3-pip python3-setuptools
sudo apt-get -y install build-essential
sudo apt-get -y install git-lfs
sudo -H pip3 install --upgrade pip
hash -d pip3
pip3 install --upgrade setuptools
pip3 install ez_setup
pip3 install Cython numpy
git lfs pull

# talib
echo "installing talib ... (you should have more then 1Gb free of ram)"
sudo apt-get -y install libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -q
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

echo "Creating venv"
python3 -m venv venv
source venv/bin/activate

pip install wheel
pip install -r requirements.txt

# point python to python3
echo "alias python=python3" >> ~/.profile
echo "alias pip=pip3" >> ~/.profile
. ~/.profile

echo "cleaning..."
sudo rm ta-lib-0.4.0-src.tar.gz && sudo rm -rf ta-lib
echo "Finished installation. "

echo "Here's the output of 'python --version' (it should be 'Python 3.x.x'):"
python --version
echo "Here's the output of 'pip --version':"
pip --version

# install Oh My Zsh
echo "installing Oh My Bash"
bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

# Install package
cd highfrek
pip install -e .