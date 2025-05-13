# to fill in the following path to evaluation!
output_model=./checkpoints/urbangpt
datapath=./data/HZMetro/HZMetro_zeroshot.json
st_data_path=../data/NYC_taxi/HZMetro_zero_pkl.pkl
res_path=./result_test/HZMetro
start_id=0
end_id=80
num_gpus=1

python ./MFP-LLM/eval/run_MFP-LLM.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
