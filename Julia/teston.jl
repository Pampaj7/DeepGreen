using Flux, CUDA, Statistics
using FileIO, Glob, ImageMagick, ImageCore

train_dir = "/home/pampaj/DeepGreen/data/cifar100_png/train"
BATCH_SIZE = 32
NUM_CLASSES = 100

# Mappa classi
classes = sort(readdir(train_dir))
class_to_idx = Dict(cls => i for (i, cls) in enumerate(classes))

# Carica solo 100 immagini in totale
function load_few(path, max_imgs=100)
    images, labels = [], Int[]
    total = 0
    for class in classes
        img_paths = glob("*.png", joinpath(path, class))
        for img_path in img_paths
            img = try
                load(img_path)
            catch
                continue
            end
            img = RGB.(img)
            img = channelview(img)
            push!(images, Float32.(img))
            push!(labels, class_to_idx[class])
            total += 1
            if total >= max_imgs
                break
            end
        end
        if total >= max_imgs
            break
        end
    end
    X = cat(images..., dims=4)
    Y = Flux.onehotbatch(labels, 1:NUM_CLASSES)
    return cu(X), cu(Y)
end

println("Loading data...")
X, Y = load_few(train_dir, 100)

# Modello minimale
model = Chain(
    Conv((3, 3), 3 => 16, pad=1), relu,
    Flux.flatten,
    Dense(32 * 32 * 16, NUM_CLASSES),
    softmax
) |> gpu

# Loss e optimizer
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM()

# Train loop semplice
for i in 1:10
    grads = gradient(Flux.params(model)) do
        loss(X, Y)
    end
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    println("Epoch $i | Loss: ", loss(X, Y))
end