# CNNs and Generative Models

These notes cover **N1–N3 and G1–G2**. The primary sources are [CNN_CV_IIIT.pptx](../Sources/CNN_CV_IIIT.pptx), especially slides 2–54, 55–64 and 84, and [Gen_models_CV_IIIT.pptx](../Sources/Gen_models_CV_IIIT.pptx), slides 2–37. They follow the teacher's confirmed CNN/generative-model headings; Vision Transformers and RNNs are excluded.

**PYQ key:** P24 denotes `CV EndSem May 24.pdf`, the one-page May 2024 CSE Computer Vision paper. Numerical examples here are constructed examples or identified lecture examples.

**Bounded interpretation:** The generative-model lecture develops GANs, so they receive the main treatment. A taxonomy slide or an older diffusion-focused paper does not establish full diffusion, VAE or adversarial-autoencoder chapters as required. CNNs are explained once here; their application to detection/tracking is developed in [Object Detection and Tracking](02_Object_Detection_and_Tracking.md).

## 1. Convolutional Neural Networks

### What a CNN is

A **convolutional neural network (CNN)** learns features from images through layers that combine local weighted sums, nonlinear activation and, often, spatial downsampling. A classifier then converts the learned representation into class scores.

```text
Image -> convolution -> activation -> pooling
      -> further feature-extraction layers
      -> flatten/aggregate -> classifier -> output probabilities
```

A small prerequisite is the neuron operation $a=\sigma(\mathbf w^T\mathbf x+b)$: weights $\mathbf w$ and bias $b$ form a weighted sum, and $\sigma$ supplies nonlinearity. A CNN organises many such operations according to the image's spatial structure.

### Important ideas

**Local connectivity:** A convolutional output depends on a local region of the previous layer rather than every input pixel. This region is its local receptive field.

**Weight sharing:** The same filter weights are reused across spatial locations. A filter can therefore recognise a similar pattern in different parts of an image without a separate set of weights for every position.

**Feature maps:** Sliding one filter over the input produces one output channel. Multiple filters learn different responses and produce multiple feature maps.

**Hierarchical representation:** Early layers often detect simple local contrast patterns; later layers combine responses over larger effective receptive fields. Two successive $3\times3$ stride-1 convolutions, without dilation, have a $5\times5$ effective receptive field in the original input. Later features need not be interpretable as individual named objects, but are learned to support the task.

**Translation behaviour:** Convolution with shared weights is translation-equivariant under appropriate boundary/stride conditions: shifting the input shifts the feature map. This is different from complete translation invariance. Pooling or spatial aggregation can reduce sensitivity to small shifts, but neither guarantees invariance to every translation.

### Learned filters versus edge filters

A fixed edge operator uses predetermined weights to respond to intensity changes. A CNN learns weights from data by minimising a task loss. Some early CNN filters resemble edge filters, but others detect colours/textures, and deeper filters combine channels into task-relevant patterns.

Both can involve convolution-like weighted sums; **the operation and the purpose of a filter are different distinctions**. Restricting every CNN layer to fixed edge kernels would prevent it from learning the richer feature combinations required by the task.

**Relevant PYQ:** [May 2024 CSE CV paper](../Papers/CV%20EndSem%20May%2024.pdf), p. 1 Q1(b), asks whether edge and convolutional filters are the same and whether edge filters should be used throughout a CNN.

**5-mark question:** Explain CNNs using local connectivity, weight sharing, feature maps and hierarchical features. Relate these ideas to learned versus fixed filters.

## 2. The Convolutional Layer

### What it computes

Let the input have shape $M\times N\times C_{\mathrm{in}}$, where $M$ is height, $N$ is width and $C_{\mathrm{in}}$ is the number of channels. An ordinary convolutional layer with $K$ filters of spatial size $F_h\times F_w$ has $K=C_{\mathrm{out}}$ output channels. Each filter spans **all input channels**.

For output channel $k$, a common CNN cross-correlation convention is

$$
Z[i,j,k]=b_k+
\sum_{a=0}^{F_h-1}\sum_{b=0}^{F_w-1}
\sum_{c=0}^{C_{\mathrm{in}}-1}
W[a,b,c,k]X[iS_h+a-P_h,jS_w+b-P_w,c].
$$

Here $i,j$ are output row/column indices, $W$ contains learned weights, $b_k$ is one bias per output channel, $S_h,S_w$ are strides and $P_h,P_w$ are symmetric padding amounts. Values outside the input are zero under zero padding. An activation usually follows this pre-activation output $Z$.

Mathematical convolution reverses the kernel; CNN libraries commonly use cross-correlation but call the layer convolution. Because the weights are learned, this convention does not change the layer's ability to learn a particular spatial filter.

### A small response calculation

For one input patch and filter,

$$
X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
W=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad b=1,
$$

the response is

$$
z=1(1)+0(2)+0(3)-1(4)+1=-2.
$$

ReLU then gives $\max(0,-2)=0$. A real multichannel filter additionally sums analogous contributions across all input channels before adding its bias.

### Stride, padding and output shape

**Stride** is the distance between neighbouring filter placements. **Padding** extends the input boundary so that filters can operate near edges and so output dimensions can be controlled.

With unit dilation and symmetric padding,

$$
M_{\mathrm{out}}=\left\lfloor\frac{M+2P_h-F_h}{S_h}\right\rfloor+1,
\qquad
N_{\mathrm{out}}=\left\lfloor\frac{N+2P_w-F_w}{S_w}\right\rfloor+1.
$$

The output shape is $M_{\mathrm{out}}\times N_{\mathrm{out}}\times K$. The floor reflects retaining full filter placements. Some classroom examples assume exact divisibility; state that assumption if using the formula without the floor. A fractional intermediate result does not universally make a convolution invalid in software.

For odd filter width $F$, stride 1 and padding $P=(F-1)/2$, spatial dimensions are preserved. Padding preserves shape, not all original information; larger filters still mix neighbouring pixels.

### Parameter count

Each filter contains $F_hF_wC_{\mathrm{in}}$ weights plus one bias, so

$$
\boxed{\#\mathrm{parameters}=(F_hF_wC_{\mathrm{in}}+1)K}.
$$

The number of spatial output positions does not multiply this count, because parameters are shared. If biases are explicitly omitted, remove the $+1$.

A $1\times1$ convolution is meaningful: each output is a weighted combination of the input channels at the same location. It mixes channels and can change channel count, although it does not directly combine neighbouring spatial locations.

### Worked shape and parameter example

Input: $32\times32\times3$. Use ten $5\times5$ filters, stride 1, padding 2.

$$
M_{\mathrm{out}}=N_{\mathrm{out}}=
\frac{32+2(2)-5}{1}+1=32.
$$

Thus the output is $32\times32\times10$. Each filter has $5\cdot5\cdot3=75$ weights and one bias, giving

$$
\#\mathrm{parameters}=76\cdot10=760.
$$

There are 10,240 scalar outputs, but only 760 learned parameters. This distinction is a common numerical-exam point.

**5-mark question:** Define channels, filters, stride and padding; calculate the output dimensions and parameter count of a convolutional layer. **Relevant PYQ:** P24 p. 1 Q4(a) asks about depth, width, stride and padding. “Depth” must be qualified: an activation volume's depth is its channel count, whereas network depth means its number of layers.

## 3. Activation, Pooling, Fully Connected and Output Layers

### 3.1 Activation functions

**What they do:** Activations introduce nonlinearity. Without nonlinearities, a sequence of linear/affine convolutional or fully connected operations would still be an affine mapping; adding more such layers would not provide the same nonlinear modelling ability.

The core CNN activation is **ReLU**:

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

It leaves positive inputs unchanged and sets negative inputs to zero. It is simple and usually avoids the positive-side saturation of sigmoid, although units that remain negative can stop receiving useful gradients.

For a binary probability output, **sigmoid** maps a scalar logit to $(0,1)$:

$$
\sigma(z)=\frac1{1+e^{-z}}.
$$

It will also be useful for the GAN discriminator. The lecture lists further activation variants, but knowing every variant's formula is not necessary for the bounded CNN core.

### 3.2 Pooling

**What it does:** Pooling summarises local responses, commonly reducing width and height. It operates on each channel separately.

For a region $\mathcal R$ with $n$ values,

$$
\operatorname{MaxPool}(\mathcal R)=\max_{(i,j)\in\mathcal R}X[i,j],
\qquad
\operatorname{AvgPool}(\mathcal R)=\frac1n\sum_{(i,j)\in\mathcal R}X[i,j].
$$

Max pooling retains the strongest response; average pooling summarises the region's average response. Both discard some spatial detail. Ordinary max/average pooling has **zero learned parameters** and preserves the channel count.

For

$$
X=\begin{bmatrix}
1&1&2&4\\5&6&7&8\\3&2&1&0\\1&2&3&4
\end{bmatrix},
$$

a $2\times2$ max pool with stride 2 gives

$$
\begin{bmatrix}6&8\\3&4\end{bmatrix}.
$$

The upper-left average-pooled value would instead be $(1+1+5+6)/4=3.25$. A $32\times32\times10$ input pooled with this window/stride becomes $16\times16\times10$.

The spatial-size formula is the same sliding-window formula as for convolution, with the pool window in place of the filter. **Global average pooling** averages every spatial location of each feature map to one scalar per channel; this explains how spatial aggregation can replace a large flatten-and-dense input.

### 3.3 Flattening and fully connected layers

**Flattening** reshapes a feature volume into a vector without learning parameters. A $5\times5\times16$ volume becomes a vector of length 400.

A **fully connected (FC) layer** connects every input component to every output neuron:

$$
\mathbf z=W\mathbf a+\mathbf b.
$$

For $n_{\mathrm{in}}$ inputs and $n_{\mathrm{out}}$ outputs, it has

$$
n_{\mathrm{in}}n_{\mathrm{out}}+n_{\mathrm{out}}
$$

parameters. Thus an FC layer from 400 inputs to 84 outputs has $400\cdot84+84=33{,}684$ parameters. Unlike convolution, ordinary FC layers do not share a local spatial kernel.

### 3.4 Class scores and softmax

The last classifier layer produces **logits**, real-valued class scores. For $Q$ mutually exclusive classes, softmax converts logits to probabilities:

$$
p_c=\frac{e^{z_c}}{\sum_{q=1}^{Q}e^{z_q}},\qquad
\sum_{c=1}^{Q}p_c=1.
$$

The predicted class is the one with greatest probability. For logits $[\log2,0]$, the exponentials are $[2,1]$, so probabilities are $[2/3,1/3]$. Softmax generally has no independent trainable parameters; the preceding score-producing layer does.

### Minimal training principle

For the correct class $y$, cross-entropy is $L=-\log p_y$. Training adjusts weights to reduce the loss, for example by a gradient-descent update

$$
\theta\leftarrow\theta-\eta\nabla_\theta L,
$$

where $\eta>0$ is the learning rate. Backpropagation computes gradients through the layers; a full derivative derivation is not needed here. Filters are learned from the training task rather than chosen to be edge detectors in every layer.

**5-mark question:** Explain the main CNN layers with a diagram and a pooling example. **Relevant PYQ:** P24 p. 1 Q4(b) asks about fundamental CNN layers and their working. A complete answer should show convolution, nonlinear activation, pooling and the classifier/output path rather than rely on a disputed count of “four” layer names.

## 4. A Complete CNN Example: A LeNet-Style Classifier

### What the example illustrates

A LeNet-style classifier converts a small grayscale image into digit-class predictions using repeated convolution/subsampling and a final classifier. The lecture's LeNet case study is on slide 55. The table below uses a **simplified teaching variant** with ordinary convolution, fixed pooling and a softmax classifier; it is not a claim that every detail matches the original historical LeNet-5 implementation.

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | Grayscale image | $32\times32\times1$ |
| C1 | Six $5\times5$ filters, stride 1, no padding; activation | $28\times28\times6$ |
| S2 | $2\times2$ pooling, stride 2 | $14\times14\times6$ |
| C3 | Sixteen $5\times5$ filters, stride 1, no padding; activation | $10\times10\times16$ |
| S4 | $2\times2$ pooling, stride 2 | $5\times5\times16$ |
| C5 | 120 $5\times5$ filters, stride 1, no padding; activation | $1\times1\times120$ |
| Classifier | Flatten, FC 84 with activation, FC 10 | Ten logits |
| Output | Softmax | Ten class probabilities |

For C3, the spatial dimension is $14-5+1=10$, and each filter spans all six input channels in this simplified version. Its parameter count is

$$
(5\cdot5\cdot6+1)16=2{,}416.
$$

C5 spans the entire $5\times5$ spatial input and produces one value per output channel. The final ten outputs correspond to classes, not ten image locations.

The main lesson is the working sequence and changing representation: local features → reduced spatial maps → richer combined features → class scores. The exam core does not require memorising benchmark results or full tables for every architecture in the lecture.

**5-mark question:** Draw and explain a CNN classifier architecture, calculate selected output shapes, and identify which layers have learned parameters.

## 5. Generative, Discriminative and Conditional Models

### What the distinction means

A **discriminative model** learns how to predict a label from observed data. A **generative model** learns a distribution of data, or a sampling process representing that distribution, so it can generate new examples. Here $x$ denotes an image and $y$ a label or condition.

| Model | Quantity represented | Typical use |
| --- | --- | --- |
| Discriminative | $p(y\mid x)$ | Predict “cat” versus “dog” for an observed image. |
| Unconditional generative | $p(x)$, explicitly or through sampling | Generate plausible images without a supplied category. |
| Conditional generative | $p(x\mid y)$ | Generate an image consistent with a supplied label/condition. |

In the lecture's contrast, a discriminative classifier distributes probability across labels for each given input, while a data-distribution model assigns probability across possible images. A generative model may support likelihood-based outlier detection **if** it provides a useful evaluable density. An implicit generator such as a basic GAN does not automatically provide such a density or outlier score.

Bayes' rule relates the distributions:

$$
p(x\mid y)=\frac{p(y\mid x)p(x)}{p(y)},\qquad p(y)>0.
$$

This identity does not mean a trained classifier alone can readily generate images: the necessary data distribution and a sampling procedure are additional requirements.

### Explicit versus implicit modelling

An **explicit-density model** specifies or approximates an evaluable data density. An **implicit model** defines a sampling mechanism without directly providing a tractable density value for each image. A basic GAN belongs to the implicit category: sampling noise and applying its generator produces samples from its learned distribution.

### Why generative models matter in vision

They can synthesise images, provide generated training examples, and support conditional image synthesis such as filling a missing region when trained for that task. Such uses require suitable training and conditioning; a basic unconditional GAN does not automatically perform every image-editing task. Generated training data can help only if they are sufficiently diverse and relevant; plausibility alone does not guarantee an improvement in a downstream classifier.

**5-mark question:** Compare discriminative, unconditional generative and conditional generative models using distributions, examples and applications. Explain why a basic GAN is an implicit model.

## 6. Generative Adversarial Networks: Components and Working

### What a GAN is

A **generative adversarial network (GAN)** learns to generate data through competition between two networks:

- The **generator** $G_{\theta_G}$ converts a latent noise vector $\mathbf z\sim p_z$ into a synthetic sample $G_{\theta_G}(\mathbf z)$.
- The **discriminator** $D_{\theta_D}(x)$ outputs a value in $(0,1)$ interpreted as the probability that an input came from the real training data rather than the generator.

Here $\theta_G,\theta_D$ denote their parameters. A simple prior $p_z$, such as a standard normal distribution, is easy to sample. The generated samples induce a distribution $p_G$; the objective is to make $p_G$ resemble the real-data distribution $p_{\mathrm{data}}$.

```text
Noise z ---> Generator G ---> Fake image ----\
                                             > Discriminator D ---> real/fake score
Real image from training data --------------/

D learns to distinguish real from generated images.
G learns through D's feedback to make generated images look real.
```

For images, the discriminator can use CNN feature extraction to recognise spatial patterns. A generator learns a mapping from latent features to image values, expanding spatial resolution as needed. Exact generator layer designs are not specified by the supplied lecture and are not required here.

At generation time, sample a new $\mathbf z$ and use $G$; the discriminator is not required to produce the image.

### Conditional generation

For a conditional GAN, supply the condition to both networks: generate $G(\mathbf z,y)$ and discriminate $D(x,y)$. The discriminator must assess whether an image is plausible **for that condition**. For example, conditioning on a digit label should control the generated digit, while changing $\mathbf z$ should allow variations within that label. This illustrates $p(x\mid y)$ without adding a separate architecture survey.

**5-mark question:** Explain a GAN with a labelled data-flow diagram, describing the generator, discriminator, latent noise, learned distribution and sampling at inference.

## 7. GAN Objectives and Alternating Training

### 7.1 The adversarial objective

The adversarial value function is

$$
\boxed{
V(D,G)=\mathbb E_{x\sim p_{\mathrm{data}}}[\log D(x)]
+\mathbb E_{\mathbf z\sim p_z}[\log(1-D(G(\mathbf z)))]}.
$$

Training seeks $\min_{\theta_G}\max_{\theta_D}V(D,G)$. An expectation is an average under the specified distribution. The discriminator maximises this objective: real images should receive values near 1, and generated images values near 0. The generator minimises the second term, attempting to increase $D(G(\mathbf z))$ so its generated images are judged real.

The two networks do not jointly minimise the same loss with the same targets. They perform opposing tasks. The setup and modified generator objective below follow lecture slides 24–37 and the [original GAN paper](https://arxiv.org/html/1406.2661v1).

### 7.2 Discriminator loss

Using gradient **descent**, minimise the negative discriminator objective:

$$
L_D=-\frac1m\sum_{i=1}^{m}
\left[\log D(x^{(i)})+
\log\left(1-D(G(\mathbf z^{(i)}))\right)\right].
$$

The real label is 1 and the generated label is 0. Hold $G$ fixed during this step and update only $\theta_D$:

$$
\theta_D\leftarrow\theta_D-\eta_D\nabla_{\theta_D}L_D.
$$

Equivalently, perform gradient ascent on the positive discriminator objective. Mixing an ascent update with a negative loss reverses the intended optimisation, so always state the convention.

### 7.3 Generator loss and the modified objective

The minimax generator loss is

$$
L_G^{\mathrm{minimax}}=
\frac1m\sum_{i=1}^{m}\log\left(1-D(G(\mathbf z^{(i)}))\right).
$$

Early in training, the discriminator may confidently reject generated samples. The original minimax form can then provide weak generator gradients. The commonly taught **non-saturating generator loss** is

$$
\boxed{L_G^{\mathrm{NS}}=-\frac1m\sum_{i=1}^{m}
\log D(G(\mathbf z^{(i)}))}.
$$

It is equivalent to maximising $\log D(G(\mathbf z))$: generated samples receive the target “real” during the generator update. This changes the generator objective while retaining the desired distribution-matching equilibrium.

Hold $D$'s parameters fixed and update $\theta_G$ to minimise $L_G^{\mathrm{NS}}$. **Gradients still pass through the discriminator's computation to the generator.** Freezing discriminator weights does not mean removing the discriminator from the gradient path.

### 7.4 Training sequence

1. Initialise both networks.
2. Draw a minibatch of real images and a minibatch of noise vectors.
3. Generate synthetic images. Train $D$ to assign real images 1 and generated images 0, updating only $D$.
4. Draw noise vectors for a generator step. Train $G$ so that $D$ assigns its outputs high “real” probability, updating only $G$.
5. Repeat, possibly using several discriminator steps per generator step as in the lecture's pseudocode.

| Update | Parameters changed | Desired discriminator outputs |
| --- | --- | --- |
| Discriminator | $\theta_D$ | $D(x_{\mathrm{real}})\to1$, $D(G(\mathbf z))\to0$ |
| Generator | $\theta_G$ | $D(G(\mathbf z))\to1$ |

### Worked loss calculation

For one real sample and one generated sample, suppose

$$
D(x_{\mathrm{real}})=0.9,\qquad D(G(\mathbf z))=0.2.
$$

Using natural logarithms and the loss normalisation above,

$$
L_D=-\log0.9-\log0.8\approx0.1054+0.2231=0.3285,
$$

$$
L_G^{\mathrm{NS}}=-\log0.2\approx1.6094.
$$

If an improved generator raises its discriminator score to 0.8, its non-saturating loss drops to $-\log0.8\approx0.2231$. This is beneficial to $G$, although it makes discrimination of that generated sample harder for $D$.

**5-mark question:** Write the GAN objective, explain every term, and describe alternating discriminator/generator updates. Include the reason for the modified generator loss and which network is fixed in each step.

## 8. GAN Equilibrium, Limitations and Applications

### What the ideal equilibrium means

For a fixed generator, the discriminator's ideal value at a point with positive total density is

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_G(x)}.
$$

To obtain it, write the local discriminator objective as $a\log d+b\log(1-d)$, where $a=p_{\mathrm{data}}(x)$ and $b=p_G(x)$. Its derivative is

$$
\frac{a}{d}-\frac{b}{1-d}=0
\quad\Rightarrow\quad a(1-d)=bd
\quad\Rightarrow\quad d=\frac{a}{a+b}.
$$

If $p_G=p_{\mathrm{data}}$, the optimal discriminator gives $1/2$ wherever this common density is positive: it cannot distinguish the two sources. This is the ideal distribution-matching result, not a guarantee that ordinary neural-network training converges. A weak discriminator can also output values near $1/2$, so that observation alone does not establish a good generator.

### Important practical limitations

**Mode collapse** occurs when the generator produces only a small variety of outputs, failing to represent the diversity of the real distribution. **Training instability** occurs because each network's updates change the other's learning problem. A discriminator that becomes much stronger than the generator can also make generator learning difficult.

These failures explain why realistic individual samples are insufficient evidence of successful distribution learning. Both quality and variety matter; a full generative-evaluation metric catalogue is not part of the confirmed core.

GANs illustrate image synthesis and, with appropriate conditioning/training, controlled generation and image-to-image applications. A vanilla GAN does not need a target image paired with each sampled noise vector: supervision comes from distinguishing generated samples from real data. This differs from an ordinary reconstruction task that directly compares a prediction with a specific target image.

**Relevant PYQ:** P24 p. 1 Q5(b) compares an adversarial autoencoder with a GAN. The GAN working is directly covered here; the autoencoder half remains outside the narrowed required scope. The diffusion-focused [IT paper](../Papers/Computer_Vision_Exam_Paper.pdf), p. 1 Q2–Q3, does not override the supplied lecture emphasis.

**5-mark question:** Explain the ideal GAN equilibrium and discuss mode collapse and training instability. A full divergence/convergence proof is unnecessary.

## 9. Exam Revision Checklist

| Question style | Essential answer components |
| --- | --- |
| Explain CNN working | Local filters, shared weights, feature maps, nonlinearity, hierarchical representation and classifier diagram. |
| Calculate layer output/parameters | Input channels, kernel depth, padding, stride, output shape, biases and shared-parameter count. |
| Explain pooling/FC/output | A pooling example, zero pool parameters, flatten length, dense connections and softmax scores. |
| Explain a CNN architecture | The LeNet-style flow and correctly calculated intermediate dimensions. |
| Compare generative/discriminative models | $p(x)$ versus $p(y\mid x)$ versus $p(x\mid y)$, sampling, and a relevant example. |
| Explain a GAN | Generator/discriminator diagram, latent distribution, loss terms, alternating updates, sampling and limitations. |

In numerical answers, do not confuse channel depth with network depth, activation count with parameter count, or zero-parameter pooling with a learned convolution. In GAN answers, do not train the generator to make the discriminator's “fake” prediction more confident. Excluded transformer questions, unrelated architecture catalogues and specialised diffusion applications are not included in these notes.
