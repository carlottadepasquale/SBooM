#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale
#
# extract path of bin directory and after cut it to obtain BMCH package path
BINDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)
BMCH_PATH=${BINDIR%%bin} 
export BMCH_HOME=$BMCH_PATH
#set bin path to system path and package path to python path
export PATH=$BINDIR:$PATH           #directory con tutti gli eseguibili, così poi posso eseguirli da qualunque punto
export PYTHONPATH=$BMCH_PATH:$PYTHONPATH 