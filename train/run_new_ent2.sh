#!/bin/bash
wt=0.618
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 5 --clip 0.06 --rt 1 --la 0.00003 --scale_const $wt --scale_weight $wt --epochs 18 --init 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 5 --clip 0.06 --rt 1 --la 0.00003  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 5 --clip 0.06 --rt 1 --la 0.00003 --scale_const $wt --scale_weight $wt --epochs 12 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 8 --clip 0.06 --rt 1 --la 0.00003  --scale_const $wt --scale_weight $wt --epochs 16 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 8 --clip 0.06 --rt 1 --la 0.00003 --scale_const $wt --scale_weight $wt --epochs 12 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 12 --clip 0.06 --rt 1 --la 0.00005  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 12 --clip 0.06 --rt 1 --la 0.00005 --scale_const $wt --scale_weight $wt --epochs 12 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 18 --clip 0.06 --rt 1 --la 0.00006  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 18 --clip 0.06 --rt 1 --la 0.00006 --scale_const $wt --scale_weight $wt --epochs 12 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 5 --clip 0.06 --rt 0.6 --la 0.00006 --scale_const $wt --scale_weight $wt --epochs 18 --init 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 18 --clip 0.06 --rt 0.6 --la 0.00006  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 18 --clip 0.06 --rt 0.6 --la 0.00006 --scale_const $wt --scale_weight $wt --epochs 12 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 30 --clip 0.06 --rt 0.6 --la 0.00008  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 30 --clip 0.06 --rt 0.6 --la 0.00008 --scale_const $wt --scale_weight $wt --epochs 12
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 50 --clip 0.06 --rt 0.6 --la 0.00011  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 50 --clip 0.06 --rt 0.6 --la 0.00011 --scale_const $wt --scale_weight $wt --epochs 12  
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 5 --clip 0.06 --rt 0.4 --la 0.0001 --scale_const $wt --scale_weight $wt --epochs 18 --init 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 30 --clip 0.06 --rt 0.4 --la 0.0002  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 30 --clip 0.06 --rt 0.4 --la 0.0002 --scale_const $wt --scale_weight $wt --epochs 12
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.00001 --gamma 90 --clip 0.06 --rt 0.4 --la 0.0003  --scale_const $wt --scale_weight $wt --epochs 24 
"/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP_ENT.py" --lr 0.000001 --gamma 90 --clip 0.06 --rt 0.4 --la 0.0003 --scale_const $wt --scale_weight $wt --epochs 12  
