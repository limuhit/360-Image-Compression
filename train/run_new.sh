#!/bin/bash
rt_list=(1 0.6 0.4)
la_list=(0.00003 0.00005 0.00006)
for i in ${!rt_list[*]};
do(
    if [ $i -lt 1 ]
    then 
        echo $i,${rt_list[$i]},${la_list[$i]};
        "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP.py" --lr 0.0001 --clip 0.06 --rt "${rt_list[$i]}" --la "${la_list[$i]}"  --scale_const 0.618 --scale_weight 0.618 --epochs 24 --init
        "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP.py" --lr 0.00001 --clip 0.006 --rt "${rt_list[$i]}" --la "${la_list[$i]}"  --scale_const 0.618 --scale_weight 0.618  --epochs 12
    else
        "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP.py" --lr 0.0001 --clip 0.06 --rt "${rt_list[$i]}" --la "${la_list[$i]}"  --scale_const 0.618 --scale_weight 0.618 --epochs 24
        "/data1/home/csmuli/anaconda3/bin/python" "/data1/home/csmuli/SphereCMP/test/trainDDP_IMP.py" --lr 0.00001 --clip 0.006 --rt "${rt_list[$i]}" --la "${la_list[$i]}"  --scale_const 0.618 --scale_weight 0.618  --epochs 12
    fi
    
)
done