#!/bin/bash

for i in {0..185}
do 
	~/.local/bin/labelme_json_to_dataset ./"0.3 - labels"/$i.json -o ./"0.3 - labels"/$i"_"json
done
