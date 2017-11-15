#!/bin/bash


# slurm seems to loos basedir so we fix it
#BASEDIR=$(dirname "$0")
BASEDIR="/home/rramosp/augergps"

PYTHON_SCRIPT=$1
PYTHON_CODE=$2
echo "using basedir" $BASEDIR
cd $BASEDIR
if [ ! -f $PYTHON_SCRIPT ]; then
    echo "python script not found:" $PYTHON_SCRIPT
    exit -1
fi
ipython2 $PYTHON_SCRIPT "$PYTHON_CODE"
