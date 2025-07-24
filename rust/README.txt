export LIBTORCH=~/libtorch-cu118/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

cargo run --release --bin resnet_cifar100

settare variabili d'ambiente tutte le volte sennò si fa casino