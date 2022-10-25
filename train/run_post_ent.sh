#!/bin/bash
for midx in {0..8}
do(
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_POST_ENT.py" --midx $midx --lr 0.001  --epochs 50 
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_POST_ENT.py" --midx $midx --lr 0.0001  --epochs 40 
    "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_POST_ENT.py" --midx $midx --lr 0.00001  --epochs 30 
)
done
    