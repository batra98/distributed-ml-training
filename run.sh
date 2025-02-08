#!/bin/bash

if [ $# -ne 1 ]; then
	echo "Usage: $0 <part_number>"
	echo "Example: $0 1"
	exit 1
fi

PART_NUMBER=$1

case $PART_NUMBER in
1)
	echo "Running Part 1: Training Model for 40 iterations (discard first iteration)"
	python task_1/main.py
	;;
*)
	echo "Invalid part number: $PART_NUMBER"
	echo "Valid parts: 1, 2, 3"
	exit 1
	;;
esac
