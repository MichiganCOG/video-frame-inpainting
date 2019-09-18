#!/bin/bash

if [ ! -d "${1}/Imagenet-VID" ]
then
  mkdir -p ${1}/Imagenet-VID/
fi

cd ${1}/Imagenet-VID/

wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ILSVRC2015_VID.tar.gz ILSVRC2015/Data/VID/test
mkdir -p test
for f in ILSVRC2015/Data/VID/test/*; do
    ffmpeg -i $f/%06d.JPEG -c:v copy test/$(basename $f).mkv -y
done

rm -rf ILSVRC2015 ILSVRC2015_VID.tar.gz
