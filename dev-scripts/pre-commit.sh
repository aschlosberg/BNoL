#!/bin/bash

set -eu

FULLPATH=`readlink -f $0`
DIR=`dirname ${FULLPATH}`

# See http://codeinthehole.com/writing/tips-for-using-a-git-pre-commit-hook/

git stash -q --keep-index # hide files not to be committed before running tests
${DIR}/run-tests.sh
PASSED=$?
git stash pop -q 2>/dev/null # return hidden files

[ $PASSED -ne 0 ] && exit 1 # doesn't make sense to do this right now but we may introduce other tests later; BASH logical ORs are ugly
exit 0
