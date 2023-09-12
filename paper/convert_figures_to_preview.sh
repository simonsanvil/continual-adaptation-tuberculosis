#!/bin/bash

mkdir -p prev_figures

for f in ./full_figures/*;
do
  echo $f
  g=${f#./full_}
  #   check if pdf
  istex=$(echo $g | grep -c ".tex")
  if [ $istex -eq 0 ]; then
    convert -density 92 $f ./prev_$g # this only works for pdfs
  else
    cp $f ./prev_$g
  fi
done