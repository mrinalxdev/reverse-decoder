# Reversible Decoder

### My chain of thoughts

so it all started with the fundamental question : "What if we could make a neural network forget things ??" I started digging deeper into this topic of reversibility in neural architectures, and it struck me that traditional transformers are one way streets. Once a token is generated, there no way to literally "go back" becuase the forward pass destroys information through non linear activations sampling. But what if we could preserve all the intermediate states and like reverse the token generation process.

This led me down the rabbut hole of invertible neural networks and RevNets (reversible residual networks), but those architectures were designed for discrimative tasks. The real challenge for me is to find out how can I maintain a "perfect" kind reversing process while preserving the autoregressive nature of language generation.