python full_GPCN_new_data.py --model_type GPCN --data twitch-eDE --lr 0.01 --hidden 512 --weight_decay 0.001 --res_scale 16 --T 2 --dropout 0.0 --mlpX 1
python full_GPCN_new_data.py --model_type GPCN --data yelp-chi --lr 0.01 --hidden 64 --weight_decay 1e-05 --res_scale 0.0625 --T 8 --dropout 0.0 --mlpX 4
python full_GPCN_new_data.py --model_type GPCN --data genius --lr 0.01 --hidden 64 --weight_decay 1e-05 --res_scale 0.25 --T 8 --dropout 0.0 --mlpX 1
python full_GPCN_new_data.py --model_type GPCN --data fb100Penn94 --lr 0.05 --hidden 512 --weight_decay 1e-05 --res_scale 0.0625 --T 4 --dropout 0.0 --mlpX 1
python full_GPCN_new_data.py --model_type GPCN --data deezer-europe --lr 0.01 --hidden 64 --weight_decay 1e-05 --res_scale 0.0625 --T 16 --dropout 0.0 --mlpX 5
python full_GPCN_new_data.py --model_type GPCN_LINK --data deezer-europe --lr 0.01 --hidden 512 --weight_decay 0.001 --mlpX  8 --res_scale 3.0  --T 1 --dropout 0.0 
python full_GPCN_new_data.py --model_type GPCN_LINK --data twitch-eDE --lr 0.01 --hidden 64  --weight_decay 0.001 --mlpX 2 --res_scale 64 --T 2  --dropout 0.0 
python full_GPCN_new_data.py --model_type GPCN_LINK --data fb100Penn94 --lr 0.01 --hidden 512 --weight_decay 1e-5 --mlpX 1 --res_scale 4 --T 1 --dropout 0.0
python full_GPCN_new_data.py --model_type GPCN_LINK --data yelp-chi --lr 0.01 --hidden 512 --weight_decay 1e-5 --mlpX 5 --res_scale 0.015625  --T 2  --dropout 0.0
python full_GPCN_new_data.py --model_type GPCN_LINK --data genius --lr 0.01  --hidden 64  --weight_decay 0.001 --mlpX 2 --res_scale 1  --T 8  --dropout 0.0 
python full_GPCN_new_data.py --model_type AGPCN --data yelp-chi --lr 0.01 --hidden 512 --weight_decay 1e-05 --mlpX 2 --T 2 --dropout 0.0 
python full_GPCN_new_data.py --model_type AGPCN --data twitch-eDE --lr 0.01 --hidden 512 --weight_decay 1e-05 --mlpX 1 --T 4 --dropout 0.0 
python full_GPCN_new_data.py --model_type AGPCN --data fb100Penn94 --lr 0.01 --hidden 64 --weight_decay 1e-05 --mlpX 2 --T 8 --dropout 0.3
python full_GPCN_new_data.py --model_type AGPCN --data deezer-europe --lr 0.01 --hidden 64 --weight_decay 1e-05 --mlpX 2 --T 1 --dropout 0.6
python full_GPCN_new_data.py --model_type AGPCN --data genius --lr 0.01  --hidden 512  --weight_decay 1e-5 --mlpX 2 --T 2 --dropout  0.0
python full_GPCN_new_data.py --model_type AGPCN_LINK --data twitch-eDE --lr 0.01 --hidden 64 --weight_decay 0.001 --mlpX 2 --T 4 --dropout 0.0
python full_GPCN_new_data.py --model_type AGPCN_LINK --data yelp-chi --lr 0.01 --hidden 512 --weight_decay 1e-05 --mlpX 3 --T 2 --dropout 0.0 
python full_GPCN_new_data.py --model_type AGPCN_LINK --data genius --lr 0.01 --hidden 512 --weight_decay 1e-05 --mlpX 3 --T 4 --dropout 0.0 
python full_GPCN_new_data.py --model_type AGPCN_LINK --data fb100Penn94 --lr 0.01 --hidden 64 --weight_decay 1e-05 --mlpX 2 --T 4 --dropout 0.3
python full_GPCN_new_data.py --model_type AGPCN_LINK --data deezer-europe --lr 0.01 --hidden 512 --weight_decay 1e-05 --mlpX 2 --T 1 --dropout 0.6






