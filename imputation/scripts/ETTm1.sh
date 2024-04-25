gpu=0
seq_len=96
root_path_name=/stage/irsa-staff-jkrick/TOTEM/imputation/data/
#data_path_name=ETTm1.csv
#data_name=ETTm1
data_path_name=ECO2_ecog_data.nc
data_name=ECO2_ecog
random_seed=2021
pred_len=0

python -u imputation/save_notrevin_notrevinmasked_revinx_revinxmasked.py\
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7\
  --gpu $gpu\
  --save_path "imputation/data/ETTm1"
