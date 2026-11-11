# DeepGenModels

Deep generative models in Julia / Flux: autoregressive models (MADE, NADE, ConvNADE, PixelCNN, WaveNet) on binarised MNIST.

## Layout

```
setup.jl              packages, the compute device, and the shared code below
src/
  training.jl         fit! (gradient descent on the NLL), logpdf over a data loader, seeds
  data.jl             binarised MNIST loaders (fixed benchmark; labelled MNIST, optionally binarised afresh every epoch)
  plotting.jl         sample mosaics, loss curves, sample grids for every checkpoint
  checks.jl           autoregressive-property checks with autograd
  toy_data.jl         2D toy data sets (moons, circles, ring, spiral, ribbon)
  made.jl             MADE
  nade.jl             DeepNADE and RNADE
  convnade.jl         ConvNADE
  pixelcnn.jl         PixelCNN: single stack, gated two stacks, class-conditional
  wavenet.jl          PixelCNN1D, WaveNet, class-conditional WaveNet
experiments/          one script per model family, run cell by cell
```

The notebooks (`*.ipynb`) and the old `utils.jl`, `io_utils.jl` and `generate_data.jl` are kept unchanged for now.

## Running an experiment

1. Put `binarized_mnist.npz` in `datasets/`
   (from <https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz>).
2. Open e.g. `experiments/pixelcnn.jl` in VS Code and start the Julia REPL.
3. Each cell starts at a line beginning with `##`. **Alt+Enter** runs the current cell; results appear inline,
   and errors point to the exact file and line.

Every script starts by moving to the repository root and including `setup.jl` and its model file, so it runs
from any working directory. Checkpoints go to `checkpoints/<model name and settings>/`, or to
`checkpoints/<checkpoint_folder>/` when `fit!(...; checkpoint_folder="...")` is given.

To pick a GPU, set `gpu_id` at the top of `setup.jl`.

### Editing the source while experimenting

With [Revise.jl](https://github.com/timholy/Revise.jl), edits to a model file take effect without restarting:
add `using Revise` to `~/.julia/config/startup.jl`, and in the setup cell load the model file you are editing
with `includet` instead of `include`, e.g. `includet("../src/pixelcnn.jl")`.
(Changing the definition of a `struct` still needs a restart.)

## Checking the code

`test/smoke_test.jl` builds every model, trains it for one short epoch on tiny fake data, and checks the
log-likelihoods, the samples and causality (about 5 minutes on a CPU; it runs in a temporary folder):

```
julia --project test/smoke_test.jl
```
