# -lam 0.01 -mu 0.1 for 4-layer CNN and 3-layer MLP
# -lam 0.0001 -mu 0.0 for ResNet-18 and fastText
# -lam 0.01 -mu 1.0 for HAR-CNN

# ---- データ生成(初回のみ) ----
# cd ../dataset
# uv run python generate_dataset.py --name mnist    --num_clients 20 --alpha 0.1
# uv run python generate_dataset.py --name fmnist   --num_clients 20 --alpha 0.1
# uv run python generate_dataset.py --name Cifar100 --num_clients 20 --alpha 0.1
# cd ../system

# ---- MNIST ----
nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 10  -data mnist-0.1-npz    -m cnn -algo GPFL -did 6 -lam 0.01 -mu 0.1 > result-mnist-0.1-npz.out    2>&1 &

# ---- Fashion-MNIST ("mnist"を含むので main.py が 1ch/dim=1024 に分岐) ----
# nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 10  -data fmnist-0.1-npz   -m cnn -algo GPFL -did 6 -lam 0.01 -mu 0.1 > result-fmnist-0.1-npz.out   2>&1 &

# ---- CIFAR-100 ("Cifar"を含むので main.py が 3ch/dim=1600 に分岐) ----
# nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 100 -data Cifar100-0.1-npz -m cnn -algo GPFL -did 6 -lam 0.01 -mu 0.1 > result-Cifar100-0.1-npz.out 2>&1 &
