using ImageMagick
using Glob
using ImageCore
using Flux
using CUDA
using Statistics
using Base.Threads
using Flux: onehotbatch
using Base.Iterators: partition
# Settings
train_dir = "/home/pampaj/DeepGreen/data/cifar100_png/train"
test_dir = "/home/pampaj/DeepGreen/data/cifar100_png/test"
batch_size = 128
img_size = (32, 32)
num_classes = 100

# Mapping class -> label index
classes = sort(readdir(train_dir))
class_to_idx = Dict(cls => i for (i, cls) in enumerate(classes))

# Conta numero immagini (serve per preallocare)
function count_images(path)
    count = 0
    for class in readdir(path)
        count += length(glob("*.png", joinpath(path, class)))
    end
    return count
end

# Fast loader
function load_data(path)
    total = count_images(path)
    X = Array{Float32}(undef, 32, 32, 3, total)
    Y = Vector{Int}(undef, total)

    i = Threads.Atomic{Int}(1)

    @threads for class in readdir(path)
        label = class_to_idx[class]
        for img_path in glob("*.png", joinpath(path, class))
            img = ImageMagick.load(img_path)
            img_f32 = Float32.(permutedims(channelview(img), (2, 3, 1)))
            idx = Threads.atomic_add!(i, 1)
            X[:, :, :, idx] = img_f32
            Y[idx] = label
        end
    end

    return X, onehotbatch(Y, 1:num_classes)
end

println("Counting and loading training data...")
train_X, train_Y = load_data(train_dir)
println("Counting and loading test data...")
test_X, test_Y = load_data(test_dir)

function make_batches(X, Y, batch_size)
    N = size(X, 4)
    idxs = partition(1:N, batch_size)
    return [(X[:, :, :, i], Y[:, i]) for i in idxs]
end

train_batches = make_batches(train_X, train_Y, batch_size)
test_batches = make_batches(test_X, test_Y, batch_size)

println("✅ Loaded ", length(train_batches), " training batches")
println("✅ Loaded ", length(test_batches), " test batches")