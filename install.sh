#!/bin/bash


cd ./models/DGUA_FAS/
sh install.sh
cd ../..

cd ./models/GACD_FAS/
sh install.sh
cd ../..

cd ./models/JPD_FAS/
sh install.sh
cd ../..



cd ./models/DGUA_FAS/
sh install.sh
cd ../..

cd ./models/GACD_FAS/
sh install.sh
cd ../..

cd ./models/JPD_FAS/
sh install.sh
cd ../..


conda clean -a -y
