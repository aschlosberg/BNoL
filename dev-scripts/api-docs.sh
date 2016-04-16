#!/bin/bash

set -eu

FULLPATH=`readlink -f $0`
DIR=`dirname ${FULLPATH}`

sphinx-apidoc -f -o ${DIR}/../docs/source ${DIR}/../bnol
exit $?
