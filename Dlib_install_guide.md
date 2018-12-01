# Dlib Installation Guide without Anaconda

## First of all you need to install Cmake and add it to Path
* Download Cmake from https://cmake.org/download/ (.zip)
* Extract and make copy to Program Files folder 
* Add to path variable c:\Program Files(x86)\Cmake....\bin
* Open Command Prompt and type 

    ` cmake --version` . 
    
    If the version of Cmake is displayed, it was installed succesfully.

## Install Dlib via pip
These instructions is for Python 3.6.4 (64bit) on Windows 10 64 bit version, please check your python version before installation.
* Download .whl file from http://pypi.fcio.net/simple/dlib/ (Make sure that dlib is a support wheel for your python and system) For me, I selectecd dlib-19.8.1-cp36-cp36m-win_amd64.whl file.
* Open Command Prompt and type this command.

    ` pip install 'PATH of wheel file' `

