# Self-Organising-Maps-SOM
A self-organizing map (SOM) or self-organising feature map (SOFM) is a type of artificial neural network (ANN) that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map. Self-organizing maps are different from other artificial neural networks as they apply competitive learning as opposed to error-correction learning (such as backpropagation with gradient descent), and in the sense that they use a neighborhood function to preserve the topological properties of the input space. (Credit: https://en.wikipedia.org/wiki/Self-organizing_map)

This code implements python classes for SOM and demonstrates it on 3-D data (e.g. normalized RGB values). It basically learns vector quantization which might especially be useful for clustering purposes. This implementation can take care of higher dimensional inputs rather than only 3. 

In single .py file, you can find the class definitions and the methods as well as data generation and visualisation utilities.
