# RObust frequency-based Structure Extraction (ROSE)

## Introduction

ROSE is a method of finding structure in cluttered 2D maps using DFT. The approach relies on the idea that repeating
structures in the map will have corresponding strong response in frequency spectrum. Then through building a appropriate
filter it is possible to suppress noise and clutter in the map.

<p align="center">
  <img src="https://github.com/tkucner/rose/blob/master/images/outline.png">
</p>

## Simple working exmaple

`python ROSE.py test_map.json input_schema.json`