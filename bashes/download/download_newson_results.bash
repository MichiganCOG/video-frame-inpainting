#!/bin/bash

wget -O newson_results.tar.gz.aa https://umich.box.com/shared/static/35ghqqdwbgav4fhhoehba0bgq7es4ida.aa
wget -O newson_results.tar.gz.ab https://umich.box.com/shared/static/b4l6gg5invad2yo9mhz6y008nfw2yhkj.ab
cat newson_results.tar.gz.aa newson_results.tar.gz.ab | tar -xz
rm newson_results.tar.gz.aa newson_results.tar.gz.ab
