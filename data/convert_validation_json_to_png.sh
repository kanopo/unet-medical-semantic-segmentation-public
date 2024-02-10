
#!/bin/bash

# for i in {0..185}
# do 
# 	labelme_json_to_dataset ./"0.4 - validation"/$i.json -o ./"0.4 - validation"/$i"_"json
# done

for entry in "./0.4 - validation"/*.json
do
  echo "$entry"
  labelme_json_to_dataset "$entry" -o "$entry""_json"
done
