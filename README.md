## Exploring the Benefits of Learning to Mask for Task Transfer

This is the code base that I used for my independent study with Emma Strubell. The goal of the project was to explore how far we can push the limits of parameter-efficient tuning. The literature in parameter efficient tuning suggests that we only need a small perturbation from the pretrained model to solve a downstream task.(e.g. enforcing an update matrix to extremely low-rank yields comparable results). In this work, we work we take one step further: Can we approximate the finetuning procedure **without changing parameters at all**?

The idea is that we can learn the binary mask such that the masked pretrained model will approximate the finetuned model. The benefit of having a binary mask instead of altered weight is that we can store the learned model in an efficient manner and run inference at a lower latency (with hardware support). Our positive finding include
* The mask learning approach performs comparably to finetuning in single-task setup
* Maksed model suffers less from negative transfer in the context of intermediate task training.
* Using (diagonal) Fisher Information Matrix to initialize the binary mask does not result in faster convergence.

One can find the detailed report in this [link](https://drive.google.com/file/d/1pEX5DJxvsgcU06FZofjLZgIrat2FdHA9/view). 
