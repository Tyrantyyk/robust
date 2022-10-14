#CUDA_VISIBLE_DEVICES=0 python threestage.py --batch_size 256 --noise_type noisy100 --num_epochs 200 --lr 0.02 --cosine \
#--dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.95 --pretrain_ep 25 \
#--start_expand 120 --data_path ../data/cifar-100-python --noise_path ../data/cifar100N/CIFAR-100_human.pt

# contest model
CUDA_VISIBLE_DEVICES=2 python contest_model.py --batch_size 256 --noise_type noisy100 --num_epochs 150 --lr 0.05 --cosine \
--dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.95 --pretrain_ep 15 \
--start_expand 60 --data_path ../data/cifar-100-python --noise_path ../data/cifar100N/CIFAR-100_human.pt

