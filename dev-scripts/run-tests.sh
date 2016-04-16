#!/bin/bash

set -eu

FULLPATH=`readlink -f $0`
DIR=`dirname ${FULLPATH}`

pyclean ${DIR}/../

exit `python -m unittest discover ${DIR}/../tests "*.py" -v`
