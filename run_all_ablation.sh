# study the dataset size
# ./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.5 --alpha2 0.5 --seed 1 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.5 --alpha2 0.5 --seed 0 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.5 --alpha2 0.5 --seed 2 --fixed_alpha

# study the dataset size
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.25 --alpha2 0.75 --seed 1 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.25 --alpha2 0.75 --seed 0 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.25 --alpha2 0.75 --seed 2 --fixed_alpha

# study the dataset size
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.75 --alpha2 0.25 --seed 1 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.75 --alpha2 0.25 --seed 0 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.75 --alpha2 0.25 --seed 2 --fixed_alpha

# study the dataset size
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.0 --alpha2 1.0 --seed 1 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.0 --alpha2 1.0 --seed 0 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 0.0 --alpha2 1.0 --seed 2 --fixed_alpha

./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 1.0 --alpha2 0.0 --seed 1 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 1.0 --alpha2 0.0 --seed 0 --fixed_alpha
./run.sh --dataset cifar100 --num_epochs 20 --ds_percentage 0.1 --grid 5 --degree 3 --noise 0.0 --classifier gnu --activation silu --alpha1 1.0 --alpha2 0.0 --seed 2 --fixed_alpha