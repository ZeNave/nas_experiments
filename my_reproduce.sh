python my_search.py --dataset cifar10 --trainval --data_loc '../datasets/cifar10' --n_runs $1 --n_samples 10 --api_loc '../datasets/NAS-Bench-201-v1_1-096897.pth' --batch_size 256 --number_of_datasets 16 --n_hidden_layers 3 --size_of_hidden_layers 512 --convolution 1 --pca 0
# python my_search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 10 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'

python my_search.py --dataset cifar10 --trainval --data_loc '../datasets/cifar10' --n_runs $1 --n_samples 100 --api_loc '../datasets/NAS-Bench-201-v1_1-096897.pth' --batch_size 256 --number_of_datasets 16 --n_hidden_layers 3 --size_of_hidden_layers 512 --convolution 1 --pca 0
# python my_search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 100 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'

python my_search.py --dataset cifar10 --trainval --data_loc '../datasets/cifar10' --n_runs $1 --n_samples 500 --api_loc '../datasets/NAS-Bench-201-v1_1-096897.pth' --batch_size 256 --number_of_datasets 16 --n_hidden_layers 3 --size_of_hidden_layers 512 --convolution 1 --pca 0
# python my_search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 500 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'

python my_search.py --dataset cifar10 --trainval --data_loc '../datasets/cifar10' --n_runs $1 --n_samples 1000 --api_loc '../datasets/NAS-Bench-201-v1_1-096897.pth' --batch_size 256 --number_of_datasets 16 --n_hidden_layers 3 --size_of_hidden_layers 512 --convolution 1 --pca 0
# python my_search.py --dataset cifar10 --trainval --data_loc './datasets/cifar10' --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset cifar100 --data_loc './datasets/cifar100'          --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'
# python my_search.py --dataset ImageNet16-120 --data_loc './datasets/ImageNet16'  --n_runs $1 --n_samples 1000 --api_loc './datasets/NAS-Bench-201-v1_1-096897.pth'

python process_results.py --n_runs $1