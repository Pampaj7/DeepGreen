using Metalhead, Flux, CUDA, Statistics

# --- Model ---
device = CUDA.functional() ? gpu : cpu
println("Using device: ", device === gpu ? "GPU" : "CPU")

model = Metalhead.resnet(Metalhead.basicblock, [2, 2, 2, 2]; inchannels=3, nclasses=100)
model = device(model)

# --- Data loading ---
# caricato da cifar.jl
include("cifar.jl")  # questo deve definire `train_batches`, `test_batches`

# --- Loss and Optimizer ---
loss(x, y) = Flux.logitcrossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))

opt = Flux.setup(Adam(1e-4), model)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

epochs = 10  # o quanto vuoi
for epoch in 1:epochs
    println("Epoch $epoch")
    
    Flux.train!(loss, Flux.params(model), train_batches, opt)
    
    push!(train_losses, sum([loss(x, y) for (x, y) in train_batches]) / length(train_batches))
    push!(test_losses, sum([loss(x, y) for (x, y) in test_batches]) / length(test_batches))
    
    push!(train_accs, mean([accuracy(x, y) for (x, y) in train_batches]))
    push!(test_accs, mean([accuracy(x, y) for (x, y) in test_batches]))

    println("  Train Loss: ", train_losses[end])
    println("  Train Acc:  ", train_accs[end])
    println("  Test Loss:  ", test_losses[end])
    println("  Test Acc:   ", test_accs[end])
end