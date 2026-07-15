mode=single
dataset=cifar100
num_epochs=100
batch_size=32
learning_rate=2e-3
weight_decay=1e-3
classifier=gnu
alpha1=0.5
alpha2=0.5
activation=silu
ds_percentage=0.1
grid=10
degree=3
noise=0.0
seed=2
trainable_activation=prelu

for model in resnet18 resnet34 vgg16 densenet121 densenet169 densenet201 mobilenet_v2 googlenet
do 
    for seed in 0 1 2
    do
        echo "Running model: $model with classifier: $classifier"
        ./run_single.sh \
            --mode $mode \
            --dataset $dataset \
            --model $model \
            --num_epochs $num_epochs \
            --batch_size $batch_size \
            --weight_decay $weight_decay \
            --seed $seed \
            --alpha1 $alpha1 \
            --alpha2 $alpha2 \
            --activation $activation \
            --ds_percentage $ds_percentage \
            --grid $grid \
            --degree $degree \
            --noise $noise \
            --learning_rate $learning_rate \
            --classifier $classifier \
            --trainable_activation $trainable_activation
    done
done



