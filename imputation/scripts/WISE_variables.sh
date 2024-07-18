
gpu=0
seq_len=15  #length of time array
root_path_name=/stage/irsa-staff-jkrick/TOTEM/imputation/data/
 #data_path_name=ETTm1.csv
 #data_name=ETTm1
 #data_path_name=ECO2_ecog_data.nc
 #data_name=ECO2_ecog
data_path_name=X_np_W1_benchmark.npy
data_name=WISE_variables
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
  --enc_in 1\ 
#  --gpu $gpu\
#  --save_path "imputation/data/WISE_variables"

for seed in 2021
do
for mask_ratio_test in 0.05
do
python imputation/imputation_performance.py \
  --dataset WISE_variables \
  --trained_vqvae_model_path "/stage/irsa-staff-jkrick/TOTEM/imputation/saved_models/final_model.pth" \
  --compression_factor 4 \
  --gpu 0 \
  --base_path "imputation/data/" \
  --mask_ratio $mask_ratio_test
done
done
