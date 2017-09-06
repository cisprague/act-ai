#!/bin/bash

cd ..

if [ ! -d "bsp" ]
then
  echo "Creating bsp directory.."
  mkdir bsp
else
  echo "Bsp directory found."
fi

cd bsp

if [ ! -f "de431a.bsp" ]
then
  echo "Downloading de431a ephemeris.."
  wget -O de431a.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de431_part-1.bsp
else
  echo "Found de431a ephemeris."
fi

if [ ! -f "de431b.bsp" ]
then
  echo "Downloading de431b ephemeris.."
  wget -O de431b.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de431_part-2.bsp
else
  echo "Found de431b ephemeris."
fi

cd ../sh
