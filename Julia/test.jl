using Metalhead
using Flux
using CUDA

device = CUDA.functional() ? gpu : cpu
println("Using device: ", device === gpu ? "GPU" : "CPU")

#PORCOOO=O=O=O=O=O=O=O=O=O=O=
model = Metalhead.resnet(Metalhead.basicblock, [2, 2, 2, 2]; inchannels=3, nclasses=100)
model = device(model)

x = rand(Float32, 32, 32, 3, 16) |> device
y = model(x)

println("Output shape: ", size(y))