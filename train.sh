BS=(8 32 32)
GA=(1 1 4)
len=${#BS[@]}
for lr in 5e-5 3e-5 1e-5; do
    for ((i=0; i<$len; i++)); do
        bs=${BS[i]}
        ga=${GA[i]}
        CUDA_VISIBLE_DEVICES=0 python train.py --lr $lr --bs $bs --ga $ga --optim adamw_torch &
        CUDA_VISIBLE_DEVICES=1 python train.py --lr $lr --bs $bs --ga $ga --optim lion_32bit &
        CUDA_VISIBLE_DEVICES=2 python train.py --lr $lr --bs $bs --ga $ga --optim sgd &
        CUDA_VISIBLE_DEVICES=3 python train.py --lr $lr --bs $bs --ga $ga --optim rmsprop &
        CUDA_VISIBLE_DEVICES=4 python train.py --lr $lr --bs $bs --ga $ga --optim adamw_torch --optim_args "weight_decay=0.1" &
        CUDA_VISIBLE_DEVICES=5 python train.py --lr $lr --bs $bs --ga $ga --optim lion_32bit --optim_args "weight_decay=0.1" &
        CUDA_VISIBLE_DEVICES=6 python train.py --lr $lr --bs $bs --ga $ga --optim sgd --optim_args "weight_decay=0.1" &
        CUDA_VISIBLE_DEVICES=7 python train.py --lr $lr --bs $bs --ga $ga --optim rmsprop --optim_args "weight_decay=0.1" &
        wait
    done
done

