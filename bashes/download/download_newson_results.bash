#!/bin/bash

wget -O newson_results.tar.gz.aa https://web.eecs.umich.edu/~szetor/media/bi-TAI-pami/newson_results.tar.gz.aa
wget -O newson_results.tar.gz.ab https://web.eecs.umich.edu/~szetor/media/bi-TAI-pami/newson_results.tar.gz.ab
cat newson_results.tar.gz.aa newson_results.tar.gz.ab | tar -xz
rm newson_results.tar.gz.aa newson_results.tar.gz.ab
