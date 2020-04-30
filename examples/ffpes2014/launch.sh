#!/bin/bash
IPI=/ssd/i-pi/bin/i-pi

rm -rf results
mkdir results
cd results
cp ../ch4oh.xml xml

$IPI xml
