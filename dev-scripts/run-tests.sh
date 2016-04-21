#!/bin/bash

set -eu

FULLPATH=`readlink -f $0`
DIR=`dirname ${FULLPATH}`

pyclean ${DIR}/../

exitCode=0

function runTestsWith {
    $($1 -m unittest discover ${DIR}/../tests '*.py' -v)
    [ $? -eq 0 ] || exitCode=1
}

if [ -z ${TRAVIS+x} ]
then
    echo "Not in TRAVIS CI Environment"
    runTestsWith "python2"
    runTestsWith "python3"
else
    echo "TRAVIS CI Environment"
    python --version
    runTestsWith "python"
fi

exit $exitCode
