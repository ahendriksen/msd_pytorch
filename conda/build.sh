#!/bin/sh

echo "------------------------------------------------------------"
echo "Build msd-pytorch Environment: "
echo "------------------------------------------------------------"

env

echo "------------------------------------------------------------"

$PYTHON setup.py clean
$PYTHON setup.py install --single-version-externally-managed --record record.txt || exit 1
