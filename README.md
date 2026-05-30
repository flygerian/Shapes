# Shapes

A tensor computation library with automatic differentiation, built from scratch in C. I started working on this because I was trying to understand deep learning from the metal up with out all the abstractions that something like pytorch and tensorflow offer.
This also has the benefit of helping me improve my skills with systems programming. I also want it to be ergonomic (well as ergonomic as you can get using C).

This is not the most performant thing in the world (there's lots of other places you can go for that). Its just an honest to God recreation of a tensor library from the mind of someone who is by no means an expert. In that vein expect to see some cruft, a lot of imperfections, a few deleted claude.md and agent.md (removed mostly because they became a crutch) and some inconsistencies. 

I'll do more work documenting the ideas here but the quickest way to see how it works is to look at the 

- [Makemore example](./base/cmd/examples/makemore_5.c) - Which is a recreation of Andrej Karpathy's character level language model and
- [Vgg](./base/cmd/examples/vgg-10.c) style model trained on [ cifar-10 ](https://www.cs.toronto.edu/~kriz/cifar.html) 

# What I have so far

- Arena allocation so you only allocate once at the beginning your programing
- A context system for grouping resources (especially useful for deciding where you want computation to happen)
- Both training on CPU and Cuda, with hand written cuda kernels (LLMs and the  [Programming massively parrallel processors book](https://www.educate.elsevier.com/book/details/9780443439001) where of tremendous help here)
- Popular layers, like batchnorm, conv, etc (look in the [shapesnn](./base/nn) package to see)
- Saving/loading the models in HF's [Safetensors](https://huggingface.co/docs/safetensors/index) format


# Building the examples

From inside the ./base folder run

```bash
cmake -S . -B build-cuda -DSHAPES_ENABLE_CUDA=ON && cmake --build build-cuda
```

if you have an nvidia gpu or set `SHAPES_ENABLE_CUDA=OFF` of you don't have one. I have plans to add support for MLX in the future, you'll also see some apple specific code in some ops files

The build binary should be in the build folder you specify (build-cuda in this case)

## Dependencies. 

Right now this project depends on 

OpenBlas,Raylib and Clay (I plan to deprecate this) via git submodules so you might need to pull in those submodles as well


