python full_GPCN_old_data.py --model_type GPCN  --data cora --lr 0.05 --hidden 64 --mlp 1 --weight_decay 0.001 --res_scale 1 --T 2 --dropout 0.6
python full_GPCN_old_data.py --model_type GPCN  --data citeseer --lr 0.01 --hidden 512 --mlp 1 --weight_decay 0.001 --res_scale 0.25 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type GPCN  --data pubmed --lr 0.01 --hidden 64 --mlp 2 --weight_decay 0.001 --res_scale 0.0625 --T 4 --dropout 0.3
python full_GPCN_old_data.py --model_type GPCN  --data chameleon --lr 0.01 --hidden 512 --mlp 1 --weight_decay 0.001 --res_scale 0.25 --T 8 --dropout 0.3
python full_GPCN_old_data.py --model_type GPCN  --data squirrel --lr 0.05 --hidden 512 --mlp 1 --weight_decay 1e-05 --res_scale 0.0625 --T 8 --dropout 0.3
python full_GPCN_old_data.py --model_type GPCN  --data cornell --lr 0.01 --hidden 512 --mlp 2 --weight_decay 0.001 --res_scale 0.015625 --T 4 --dropout 0.3
python full_GPCN_old_data.py --model_type GPCN  --data film --lr 0.05 --hidden 512 --mlp 2 --weight_decay 0.001 --res_scale 0.0625 --T 2 --dropout 0.0
python full_GPCN_old_data.py --model_type GPCN  --data texas --lr 0.01 --hidden 512 --mlp 3 --weight_decay 0.001 --res_scale 0.015625 --T 1 --dropout 0.6
python full_GPCN_old_data.py --model_type GPCN  --data wisconsin --lr 0.05 --hidden 512 --mlp 2 --weight_decay 0.001 --res_scale 0.0625 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type GPCN-LINK  --data cora --lr 0.01 --hidden 512 --mlp 1 --weight_decay 0.001 --res_scale 4 --T 2 --dropout 0.6
python full_GPCN_old_data.py --model_type GPCN-LINK  --data citeseer --lr 0.01 --hidden 64 --mlp 1 --weight_decay 1e-05 --res_scale 64 --T 1 --dropout 0.3 
python full_GPCN_old_data.py --model_type GPCN-LINK  --data pubmed --lr 0.01 --hidden 512 --mlp 3 --weight_decay 0.001 --res_scale 0.0625 --T 4 --dropout 0.3 
python full_GPCN_old_data.py --model_type GPCN-LINK  --data cornell --lr 0.01 --hidden 512 --mlp 4 --weight_decay 0.001 --res_scale 0.015625 --T 1 --dropout 0.0 
python full_GPCN_old_data.py --model_type GPCN-LINK  --data chameleon --lr 0.01 --hidden 512 --mlp 2 --weight_decay 0.001 --res_scale 0.00390625 --T 4 --dropout 0.6 
python full_GPCN_old_data.py --model_type GPCN-LINK  --data cornell --lr 0.01 --hidden 512 --mlp 4 --weight_decay 0.001 --res_scale 0.015625 --T 1 --dropout 0.0 
python full_GPCN_old_data.py --model_type GPCN-LINK  --lr 0.01 --hidden 512 --mlp 2 --weight_decay 0.001 --res_scale 0.015625 --T 8 --dropout 0.0 
python full_GPCN_old_data.py --model_type GPCN-LINK  --data texas --lr 0.01 --hidden 512 --mlp 4 --weight_decay 0.001 --res_scale 0.00390625 --T 1 --dropout 0.0 
python full_GPCN_old_data.py --model_type GPCN-LINK --data wisconsin --lr 0.01 --hidden 512 --mlp 3 --weight_decay 0.001 --res_scale 0.25 --T 2 --dropout 0.0
python full_GPCN_old_data.py --model_type GPCN-LINK --data squirrel --lr 0.05 --hidden 512 --mlp 1 --weight_decay 1e-05 --res_scale 4 --T 8 --dropout 0.0 
python full_GPCN_old_data.py --model_type AGPCN --data cora --lr 0.05 --hidden 512 --mlp 1 --weight_decay 0.001 --T 2 --dropout 0.6
python full_GPCN_old_data.py --model_type AGPCN --data citeseer --lr 0.01 --hidden 512 --mlp 1 --weight_decay 0.001 --T 2 --dropout 0.6
python full_GPCN_old_data.py --model_type AGPCN --data pubmed --lr 0.01 --hidden 64 --mlp 2 --weight_decay 0.0 --T 2 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN --data chameleon --lr 0.01 --hidden 512 --mlp 1 --weight_decay 0.001 --T 1 --dropout 0.6
python full_GPCN_old_data.py --model_type AGPCN --data squirrel --lr 0.01 --hidden 64 --mlp 1 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN --data cornell --lr 0.05 --hidden 512 --mlp 2 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN --data texas --lr 0.05 --hidden 512 --mlp 2 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN --data wisconsin --lr 0.05 --hidden 512 --mlp 2 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN --data film --lr 0.05 --hidden 64 --mlp 3 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN-LINK --data cora --lr 0.01 --hidden 64 --mlp 1 --weight_decay 0.001 --T 2 --dropout 0.6
python full_GPCN_old_data.py --model_type AGPCN-LINK --data citeseer --lr 0.01 --hidden 512 --mlp 1 --weight_decay 0.001 --T 2 --dropout 0.6
python full_GPCN_old_data.py --model_type AGPCN-LINK --data pubmed --lr 0.01 --hidden 512 --mlp 2 --weight_decay 0.001 --T 2 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN-LINK --data chameleon --lr 0.05 --hidden 64 --mlp 1 --weight_decay 0.001 --T 2 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN-LINK --data squirrel --lr 0.05 --hidden 512 --mlp 1 --weight_decay 1e-05 --T 8 --dropout 0.0
python full_GPCN_old_data.py --model_type AGPCN-LINK --data cornell --lr 0.01 --hidden 512 --mlp 4 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN-LINK --data texas --lr 0.01 --hidden 512 --mlp 4 --weight_decay 0.001 --T 1 --dropout 0.3
python full_GPCN_old_data.py --model_type AGPCN-LINK --data wisconsin --lr 0.01 --hidden 512 --mlp 2 --weight_decay 0.001 --T 2 --dropout 0.0
python full_GPCN_old_data.py --model_type AGPCN-LINK --data film --lr 0.01 --hidden 512 --mlp 2 --weight_decay 0.001 --T 8 --dropout 0.0

