#!/bin/bash

cd /opt/src

### YOU CAN USE YOUR OWN CONFIG FILE
### ALSO PATHS TO DATA CAN BE MODIFIED FROM THE CONFIG
sh run/eval.sh /export/ouput/openscene config/replica/ours_openseg_pretrained.yaml ensemble
