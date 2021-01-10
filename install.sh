#!/bin/bash

# Author: Giacomo Iadarola

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

rm -rf "${SCRIPTPATH}"/fftw-*
rm -rf "${SCRIPTPATH}"/ext_tools/lear*


function validate_url(){
  if [[ $(wget -S --spider "$1" 2>&1 | grep 'HTTP/1.1 200 OK') ]]; then
    echo "true"
  fi
}

# Check if virtualenv installed
if ! command -v virtualenv &> /dev/null
then
    echo "'virtualenv' not found, proceed with installation? [y/N]"
    # shellcheck disable=SC2162
    read -n 2 RES
    if [ "$RES" == "y" ]; then
        echo "Installing python3-virtualenv..."
        sudo apt install python3-virtualenv
    else
        echo ""
        echo "OK, proceeding without virtualenv..."
    fi
else
  echo "Create the virtual environment 'venv' with python3 in "
  virtualenv "${SCRIPTPATH}/venv" -p /usr/bin/python3

  # Activate the virtual environment
  # shellcheck disable=SC1090
  source "${SCRIPTPATH}"/venv/bin/activate
fi

echo "Install External Dependencies"
echo "Installing FFTW for GIST descriptor..."

HARDCODED_FFTW_URL="http://www.fftw.org/fftw-3.3.8.tar.gz"
FFTW_VERSION_TAR=$(basename ${HARDCODED_FFTW_URL})
# NB! Assuming the tar file has format 'fftw-x.x.x.tar.gz'
FFTW_VERSION=$(echo $FFTW_VERSION_TAR | cut -d'.' -f1-3)

if [[ $(validate_url $HARDCODED_FFTW_URL) ]]; then
  wget ${HARDCODED_FFTW_URL} -P "${SCRIPTPATH}/ext_tools"

  if [[ -f "${SCRIPTPATH}/ext_tools/${FFTW_VERSION_TAR}" ]]; then
    cd "${SCRIPTPATH}/ext_tools" || exit
    tar -zxvf "${SCRIPTPATH}/ext_tools/${FFTW_VERSION_TAR}"
    rm "${SCRIPTPATH}/ext_tools/${FFTW_VERSION_TAR}"
    cd "${SCRIPTPATH}/ext_tools/${FFTW_VERSION}" || exit
    echo "Requiring sudo privileges..."
    sudo ./configure --enable-single --enable-shared
    sudo make
    sudo make install
    cd "${SCRIPTPATH}" || exit
  else
    echo "${FFTW_VERSION_TAR} was not downloaded... exiting"
    exit
  fi

  echo "Installing GIST library from git repository and numpy as pre-requisites"
  "${SCRIPTPATH}"/venv/bin/pip install numpy
  cd "${SCRIPTPATH}"/ext_tools || exit
  git clone "https://github.com/tuttieee/lear-gist-python.git"
  cd "${SCRIPTPATH}"/ext_tools/lear-gist-python || exit
  "${SCRIPTPATH}"/ext_tools/lear-gist-python/download-lear.sh
  # NECESSARY to install python3-dev
  sudo apt-get install python3-dev
  "${SCRIPTPATH}"/venv/bin/python "${SCRIPTPATH}"/ext_tools/lear-gist-python/setup.py build_ext -I ${SCRIPTPATH}/ext_tools/${FFTW_VERSION} -L ${SCRIPTPATH}/ext_tools/${FFTW_VERSION}
  "${SCRIPTPATH}"/venv/bin/python "${SCRIPTPATH}"/ext_tools/lear-gist-python/setup.py install
  cd "${SCRIPTPATH}" || exit

else
  echo "FFTW resources does not exist anymore at ${HARDCODED_FFTW_URL}, please manually install FFTW!"
  echo "See resources at 'http://www.fftw.org/fftw3_doc/Installation-on-Unix.html'"
  exit

fi

# "${SCRIPTPATH}"/venv/bin/pip install -r "${SCRIPTPATH}"/requirements_ubuntu20.txt

echo "Setting main_path to tami in utils.config"
# tells git to ignore changes in config.py
# can be reversed with 'git update-index --no-skip-worktree <file-list>'
git update-index --skip-worktree "${SCRIPTPATH}"/utils/config.py

# Using '@' insted of '/' as delimiter for sed, to avoid confusion
STANDARD_VAR_MAIN="main_path = \"/<INSERT_FULL_PATH_TO_REPO_FOLDER>/tami/\""
NEW_VAR_MAIN="main_path = \"${SCRIPTPATH}/\""
sed -i "s@${STANDARD_VAR_MAIN}@${NEW_VAR_MAIN}@g" "${SCRIPTPATH}"/utils/config.py

echo "Setup complete!"



