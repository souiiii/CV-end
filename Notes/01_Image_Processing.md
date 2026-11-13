# Image Processing and Video Analysis

These notes cover scope items **I1–I3**: image basics, frequency-domain analysis and a bounded interpretation of time-domain analysis. The teacher's four groups remain examinable; the old question distribution is not assumed. Vision Transformers and RNNs are excluded.

**Interpretation of the uncertain heading:** Here, *time-domain analysis* means directly examining changes across video frames. The supplied material does not establish whether the teacher instead meant spatial/pixel-domain processing. Section 3 explains the spatial convolution needed to connect both domains; Section 6 covers only elementary temporal change and smoothing. This does not add a general enhancement or motion-estimation syllabus.

**Sources:** [Camera Geometry](../Sources/CameraGeometry_CV_IIIT.pptx), slides 6–9 and 49, supports acquisition and sampling; [CNN](../Sources/CNN_CV_IIIT.pptx), slide 18, supports convolution. The [syllabus](../Syllabus/CV-7th-Semester-IT.pdf), p. 2, Module I, provides context. Supplementary sources are linked where the lectures leave gaps. All numerical examples below are constructed examples unless explicitly identified as PYQs.

**PYQ key:** PIP denotes `Image_Processing_Question_Papers_Extracted.pdf`. Page numbers refer to 1-based PDF pages, not the page numbering within an extracted paper.

## 1. Digital Images and Image Acquisition

### What an image represents

A digital image is a finite array of samples representing a scene. For a grayscale image, write

$$
f[x,y],\qquad 0\le x<N,\quad 0\le y<M,
$$

where $x$ is the column index, $y$ is the row index, $N$ is the width, and $M$ is the height. Each array entry is a **pixel value**. In a grayscale image it represents brightness or intensity, rather than an object label.

An RGB image has three values at each location, corresponding to red, green and blue channels. Its shape is $M\times N\times3$. An image of shape $32\times32\times3$ has 1,024 spatial locations and 3,072 scalar channel values. A binary image uses only two possible values, whereas an 8-bit grayscale image can represent 256 values, conventionally 0–255.

The image coordinate origin is usually the upper-left corner, with $x$ increasing rightward and $y$ downward. This differs from the upward-positive Cartesian convention often used to derive rotation matrices. Always state the convention in a numerical problem.

### Digital-camera acquisition

```text
Scene light -> lens -> image on CCD/CMOS sensor
           -> electrical measurements -> sampling and quantization
           -> digital image processing -> stored/displayed image
```

The lens forms an image on a photosensitive sensor. Sensor elements integrate incident light and produce electrical measurements. Digitization assigns spatial sample locations and numerical intensity values. A camera's image processor may subsequently adjust colour and brightness before storage.

The pinhole projection used for geometric reasoning is an idealisation of this acquisition process. Its equations and camera parameters are developed in [Camera Calibration](03_Camera_Calibration.md).

### Sampling and quantization

**Sampling** makes the spatial coordinates discrete. **Quantization** makes the measured intensity discrete. They affect different aspects of an image.

If a continuous image is $f_c(\xi,\eta)$ and the sample spacings are $\Delta_x,\Delta_y$, a simple model is

$$
f[x,y]=Q\!\left(f_c(x\Delta_x,y\Delta_y)\right),
$$

where $Q$ maps the measured value to an allowed intensity level.

| Property | Meaning | Effect of insufficient resolution |
| --- | --- | --- |
| Spatial resolution | Ability to represent spatial detail, determined by sampling density together with the imaging optics | Fine structures merge, edges become jagged, and aliasing may occur. |
| Intensity resolution | Number of distinguishable intensity levels | Smooth gradients may become visible bands, called false contouring. |

With $b$ bits per channel, the number of available levels is $L=2^b$. Increasing the pixel count does not by itself increase the number of intensity levels. Increasing bit depth does not restore spatial detail lost before sampling.

### Image storage: worked example

For an uncompressed image with $C$ channels and $b$ bits per channel,

$$
\text{storage}=MNCb\text{ bits}=\frac{MNCb}{8}\text{ bytes}.
$$

For a $640\times480$ RGB image with 8 bits per channel:

$$
640\cdot480\cdot3\cdot8=7{,}372{,}800\text{ bits},
$$

$$
\text{storage}=921{,}600\text{ bytes}=900\text{ KiB}.
$$

This excludes file headers, padding and compression. An “8-bit RGB image” can be ambiguous: specify **8 bits per channel**, meaning 24 bits per pixel. A truly binary image needs one bit per pixel if packed efficiently, even if a software array stores each binary value in a byte.

**5-mark preparation:** Explain digital image formation, distinguish sampling from quantization, and calculate storage for a specified image.

## 2. Sampling, Spatial Frequency and Aliasing

### What spatial frequency means

Spatial frequency measures how rapidly an image pattern changes with position. A broad, slowly varying brightness region contains low spatial frequencies. Fine stripes and sharp transitions require high spatial frequencies. An edge generally contains a range of frequencies, not a single frequency. Noise is not necessarily confined to high frequencies.

A sinusoidal pattern can be written as

$$
f_c(\xi,\eta)=A\cos\left[2\pi(\nu_x\xi+\nu_y\eta)+\phi\right],
$$

where $A$ is amplitude, $\nu_x,\nu_y$ are frequency components in cycles per spatial unit, and $\phi$ is phase. Spatial frequency is measured in cycles per unit distance or cycles per pixel; temporal frequency is measured in cycles per second.

### Aliasing and the sampling condition

**Aliasing** occurs when different continuous patterns become indistinguishable after sampling. A fine stripe pattern may appear as coarse stripes or a moiré pattern. To reconstruct a band-limited image, the sampling rate must exceed twice the highest frequency component along each sampled axis:

$$
\frac1{\Delta_x}>2\nu_{x,\max},\qquad
\frac1{\Delta_y}>2\nu_{y,\max}.
$$

The corresponding Nyquist limits are $\Delta_x=1/(2\nu_{x,\max})$ and $\Delta_y=1/(2\nu_{y,\max})$. Use smaller spacings in practice: sampling exactly at the boundary can lose a sinusoid for particular phases. Before reducing resolution, apply a low-pass filter to suppress frequencies beyond the new sampling limit. This sampling interpretation follows the [MIT sampling chapter](https://visionbook.mit.edu/sampling_and_aliasing.html).

### Worked sampling problem

Consider the actual form used in **PIP p. 1 Q3(a)**:

$$
f_c(\xi,\eta)=4\cos\left[2\pi(\xi+2\eta)\right].
$$

The frequency components are $\nu_x=1$ and $\nu_y=2$ cycles per unit. Therefore,

$$
\Delta_x<\frac12,\qquad \Delta_y<\frac14
$$

provides a safe sampling condition. An exam answer may report $1/2$ and $1/4$ as the **maximum Nyquist intervals**, with the strict inequality caveat. The amplitude 4 does not affect the required spatial sampling rates.

**5-mark preparation:** Define aliasing, explain its cause and prevention, and determine sampling intervals for a two-dimensional sinusoid.

## 3. Spatial Filtering and Convolution

### What a spatial filter does

A spatial filter computes a new pixel value from a neighbourhood of input pixels. A **kernel** specifies the neighbourhood weights. This is the minimum spatial-domain background needed for frequency filtering and CNNs.

For an image $f$ and kernel $h$, discrete convolution is

$$
g[x,y]=(f*h)[x,y]
=\sum_a\sum_b h[a,b]f[x-a,y-b].
$$

The negative offsets imply reversal of the kernel relative to sliding cross-correlation. CNN implementations commonly use cross-correlation, $\sum h[a,b]f[x+a,y+b]$, while retaining the name “convolution.” A symmetric averaging kernel gives the same result under either convention.

### Averaging example

A $3\times3$ mean kernel is

$$
h=\frac19
\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}.
$$

For the neighbourhood

$$
\begin{bmatrix}10&10&10\\10&100&10\\10&10&10\end{bmatrix},
$$

the filtered centre becomes

$$
g=\frac{8(10)+100}{9}=20.
$$

An isolated intensity peak is reduced, but an actual small bright feature can also be blurred. The kernel sum is 1, so a constant region retains its brightness away from boundary effects.

At borders, some kernel positions fall outside the image. A problem must specify whether to use zero padding, reflection, replication, or only valid positions. Padding changes border values; it is not part of the underlying scene.

### Why an averaging filter is low-pass

For a centred $3\times3$ mean kernel, its frequency response is

$$
\mathcal H(\omega_x,\omega_y)
=\frac19(1+2\cos\omega_x)(1+2\cos\omega_y),
$$

where $\omega_x,\omega_y$ are angular spatial frequencies in radians per pixel. Each factor follows from $e^{-j\omega}+1+e^{j\omega}=1+2\cos\omega$.

At zero frequency the gain is 1. At $(\pi,0)$ the gain is $-1/3$, whose magnitude is $1/3$: rapid alternation is attenuated. A mean filter is therefore a smoothing low-pass filter, although its response is not an ideal abrupt cutoff.

**Relevant PYQ:** PIP p. 4 Q2(b) asks about mean versus weighted-mean filtering and the low-pass response of a $3\times3$ mean filter. Weighting changes the response; the derivation above covers the confirmed frequency-analysis portion.

## 4. Fourier Transform and Frequency-Domain Representation

### What the transform does

Fourier analysis represents a signal as a combination of sinusoidal components. The **discrete Fourier transform (DFT)** applies to a finite sampled signal. The **FFT** is an efficient algorithm for calculating the DFT, not a different transform.

For a length-$L$ sequence $a[n]$,

$$
A[k]=\sum_{n=0}^{L-1}a[n]e^{-j2\pi kn/L},
\qquad 0\le k<L,
$$

$$
a[n]=\frac1L\sum_{k=0}^{L-1}A[k]e^{+j2\pi kn/L},
$$

where $j^2=-1$. The forward transform uses a negative sign; the inverse uses a positive sign and the normalization $1/L$.

For an image with width $N$ and height $M$,

$$
F[k,l]=\sum_{x=0}^{N-1}\sum_{y=0}^{M-1}
f[x,y]e^{-j2\pi(kx/N+ly/M)},
$$

$$
f[x,y]=\frac1{MN}\sum_{k=0}^{N-1}\sum_{l=0}^{M-1}
F[k,l]e^{+j2\pi(kx/N+ly/M)}.
$$

Here $k$ indexes horizontal frequency and $l$ indexes vertical frequency. A 2D DFT can be computed by applying a 1D DFT to every row and then every column. These transform definitions are verified against [MIT's Fourier-analysis chapter](https://visionbook.mit.edu/image_processing_fourier.html).

### Magnitude, phase and DC

A coefficient is complex:

$$
F=\operatorname{Re}F+j\operatorname{Im}F=|F|e^{j\phi},
$$

$$
|F|=\sqrt{(\operatorname{Re}F)^2+(\operatorname{Im}F)^2},
\qquad \phi=\operatorname{atan2}(\operatorname{Im}F,\operatorname{Re}F).
$$

Magnitude describes the strength of a frequency component. Phase describes its alignment; changing phase can change where image structures occur even if the magnitude is unchanged. Both are needed for general reconstruction.

The **DC coefficient** is $F[0,0]=\sum_{x,y}f[x,y]$. The mean intensity is $F[0,0]/(MN)$. In the usual unshifted array, DC is at the upper-left. A display shift moves DC to the centre so that nearby frequencies are easy to inspect. A log display, $\log(1+|F|)$, reveals weak components alongside strong ones. Shifting the array or taking a logarithm for display is not itself a filtering operation. [OpenCV's Fourier tutorial](https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html) describes the DFT/FFT distinction and spectrum display.

For real images, coefficients have conjugate symmetry:

$$
F[(-k)\bmod N,(-l)\bmod M]=F[k,l]^*.
$$

### Worked 1D DFT

Let $a=[1,2,3,4]$, with $L=4$. The basic exponential is $e^{-j2\pi/4}=-j$.

$$
\begin{aligned}
A[0]&=1+2+3+4=10,\\
A[1]&=1+2(-j)+3(-1)+4j=-2+2j,\\
A[2]&=1-2+3-4=-2,\\
A[3]&=1+2j-3-4j=-2-2j.
\end{aligned}
$$

For example, the inverse at $n=0$ gives

$$
a[0]=\tfrac14[10+(-2+2j)-2+(-2-2j)]=1.
$$

The DC coefficient is the sum 10; the mean is $10/4=2.5$. The conjugate pair $A[1],A[3]$ is consistent with a real input.

**Relevant PYQ:** PIP p. 5 Q3(a) asks for the matrix expression of the forward 1D DFT and a numerical DFT. In matrix notation, $\mathbf A=W\mathbf a$, with $W_{kn}=e^{-j2\pi kn/L}$ and $W^{-1}=W^H/L$, where $H$ denotes conjugate transpose.

### Worked 2D DFT

Take the image

$$
f=\begin{bmatrix}1&2\\3&4\end{bmatrix}.
$$

Taking the length-2 DFT of each row yields

$$
\begin{bmatrix}1+2&1-2\\3+4&3-4\end{bmatrix}
=\begin{bmatrix}3&-1\\7&-1\end{bmatrix}.
$$

Taking the DFT of each resulting column yields

$$
F=\begin{bmatrix}10&-2\\-4&0\end{bmatrix}.
$$

In this displayed array, rows correspond to $l$ and columns to $k$. Thus $F[0,0]=10$, $F[1,0]=-2$ and $F[0,1]=-4$. The mean is $10/4=2.5$.

**5-mark preparation:** State the 2D DFT/inverse pair, explain magnitude, phase and DC, and calculate a small DFT.

## 5. Frequency-Domain Filtering

### What it is and how it works

Frequency-domain filtering modifies selected frequency components before reconstructing the image. Let $\mathcal H[k,l]$ be the filter transfer function. The basic operation is

$$
G[k,l]=\mathcal H[k,l]F[k,l],\qquad
g=\operatorname{IDFT}(G).
$$

```text
Input image -> DFT -> multiply by frequency response -> inverse DFT
                       ^
                 desired filter
```

For spatial convolution, the convolution theorem gives

$$
f*h\quad\longleftrightarrow\quad F\mathcal H.
$$

For finite arrays, multiplying equal-sized DFTs produces **circular convolution**: samples wrap around the boundaries. To calculate full linear convolution of an $M\times N$ image with a $P\times Q$ kernel, pad both to at least $(M+P-1)\times(N+Q-1)$ before transforming. Place the kernel origin consistently; an origin mistake introduces a spatial shift. The convolution relationship is discussed in the [MIT Fourier chapter](https://visionbook.mit.edu/image_processing_fourier.html).

### Low-pass and high-pass interpretation

| Filter | Frequency action | Typical image effect and limitation |
| --- | --- | --- |
| Low-pass | Retains DC/slow variation and attenuates rapid variation | Smooths images and can suppress fine-scale noise, but also blurs detail. |
| High-pass | Attenuates DC/slow variation and retains rapid variation | Emphasises transitions/fine detail, but can amplify noise and produce signed outputs. |

An illustrative ideal low-pass response is 1 inside a radius $D_0$ around the centred DC and 0 outside. A complementary high-pass is $\mathcal H_{HP}=1-\mathcal H_{LP}$. Abrupt cutoffs can cause ringing near edges. Detailed named filter-family design is outside the bounded scope.

### Worked filtering example

For the previous $2\times2$ image, retain only DC:

$$
\mathcal H=\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad
G=\begin{bmatrix}10&0\\0&0\end{bmatrix}.
$$

Every inverse-transform value is $10/4=2.5$, so

$$
g_{LP}=\begin{bmatrix}2.5&2.5\\2.5&2.5\end{bmatrix}.
$$

The complementary output is

$$
g_{HP}=f-g_{LP}
=\begin{bmatrix}-1.5&-0.5\\0.5&1.5\end{bmatrix}.
$$

Its mean is zero. Negative values are legitimate filter responses; clipping them immediately would lose information. This example uses circular filtering on the whole $2\times2$ array and illustrates mean/detail separation.

### DFT conjugation: relevant PYQ reasoning

**PIP p. 6 Q5(b)** describes centring an image with $(-1)^{x+y}$, taking its DFT, conjugating the transform, applying the inverse, and undoing the centring.

The key identity is

$$
\operatorname{IDFT}\{F^*[k,l]\}
=f^*[(-x)\bmod N,(-y)\bmod M].
$$

To see why, conjugate the forward sum and substitute it into the inverse sum. The sums over $k,l$ vanish unless the input coordinates are $-x,-y$ modulo the dimensions. For a real image, conjugation leaves the intensities unchanged and reverses both coordinates. For even-sized images, the centring factors cancel after this reversal. The result is a circular coordinate reversal, visually resembling a 180-degree rotation with the array-origin wraparound understood. Conjugation is not a blur or a brightness inversion.

**5-mark preparation:** Explain the filtering pipeline and convolution theorem, compare low-pass/high-pass results, and state why padding is necessary.

## 6. Time-Domain Analysis: Basic Temporal Video Analysis

### What it is

Temporal analysis examines how image measurements change over time. Write a sampled video as $I_t[x,y]$, where integer $t$ identifies the frame. At a fixed location, $I_0[x,y],I_1[x,y],\ldots$ forms a time sequence. Frame rate determines the temporal sampling interval: at $r$ frames per second, $\Delta t=1/r$ seconds.

This section is a **stated interpretation of the teacher's unresolved term**, not evidence of a particular taught algorithm. It covers the smallest useful video-analysis tools without adding optical-flow or recurrent-model chapters.

### Frame differencing

Frame differencing highlights changes between consecutive frames:

$$
D_t[x,y]=|I_t[x,y]-I_{t-1}[x,y]|.
$$

A threshold gives a binary change mask,

$$
B_t[x,y]=
\begin{cases}
1,&D_t[x,y]>\tau,\\
0,&\text{otherwise},
\end{cases}
$$

where $\tau$ controls sensitivity. Compute the subtraction in a signed or floating-point representation to avoid unsigned integer wraparound.

```text
Previous frame -----\
                     absolute difference -> threshold -> change mask
Current frame ------/
```

For

$$
I_{t-1}=\begin{bmatrix}10&10\\10&10\end{bmatrix},\qquad
I_t=\begin{bmatrix}10&70\\12&10\end{bmatrix},
$$

we obtain

$$
D_t=\begin{bmatrix}0&60\\2&0\end{bmatrix}.
$$

With $\tau=20$, only the upper-right location is marked. This identifies **change**, not a semantic category or a persistent object identity. A stationary object eventually disappears from a consecutive-frame difference. Camera motion, shadows and lighting changes may also produce differences.

### Temporal smoothing

A short moving average reduces frame-to-frame fluctuations:

$$
\bar I_t[x,y]=\frac1L\sum_{q=0}^{L-1}I_{t-q}[x,y].
$$

For a stable pixel with measurements $[98,102,100]$, the three-frame average is 100. For independent zero-mean noise of variance $\sigma^2$, averaging $L$ observations of an unchanged signal reduces noise variance to $\sigma^2/L$. Moving objects violate the unchanged-signal assumption and can create trails or ghosting.

A compact recursive alternative is

$$
S_t=\alpha I_t+(1-\alpha)S_{t-1},\qquad0<\alpha\le1.
$$

If $S_{t-1}=100$, $I_t=120$ and $\alpha=0.25$, then $S_t=105$. A larger $\alpha$ adapts faster but smooths less.

Comparing a frame against a maintained background image is **background subtraction**. It differs from comparing only two consecutive frames. Its basic goal is a foreground mask, as described by [OpenCV's background-subtraction tutorial](https://docs.opencv.org/3.4.9/d1/dc5/tutorial_background_subtraction.html). A full statistical background-modelling method is not required here.

### Spatial, temporal and frequency analysis compared

| Domain | Independent variable | Example operation |
| --- | --- | --- |
| Spatial | Pixel location within a frame | Average neighbouring pixels using a kernel. |
| Temporal | Frame/time at a fixed location or target | Compare successive frames or average observations over time. |
| Frequency | Sinusoidal variation rate in space or time | Attenuate high spatial-frequency coefficients. |

Temporal change detection can provide candidate regions for tracking, but tracking must additionally associate observations and preserve identity. That distinction is developed in [Object Detection and Tracking](02_Object_Detection_and_Tracking.md).

**5-mark preparation:** Define temporal analysis, derive a frame-difference mask with an example, and explain temporal smoothing and its limitations. No supplied PYQ unambiguously confirms this interpretation of “time-domain analysis.”

## 7. Exam Revision and Source Boundaries

For a 5-mark explanatory answer, give a definition, the relevant equation with symbols, a working diagram or example, and a limitation. For a numerical answer, state dimensions, transform normalization, boundary assumptions and units before calculating.

| Verified practice reference | Relevant preparation |
| --- | --- |
| [PIP](../Papers/Image_Processing_Question_Papers_Extracted.pdf), p. 1 Q1(c) | Spatial/intensity resolution. |
| PIP p. 1 Q3(a) | Sampling, aliasing and the sinusoidal-image calculation in Section 2. |
| PIP p. 5 Q1(a) | Storage calculation component only; do not infer a thresholding chapter. |
| PIP p. 4 Q2(b) | Mean-filter frequency response. |
| PIP p. 5 Q3(a) | Forward DFT, transform matrix and a numerical transform. |
| PIP p. 6 Q5(b) | Conjugation and circular spatial reversal. |

These papers include other branches; their questions guide practice rather than predict this exam. Histogram equalization/matching, detailed point-enhancement operations, adjacency problems, DCT/Haar/Hadamard transform catalogues and morphology have not been promoted into required chapters. The remaining genuine uncertainty is the teacher's intended meaning and depth of “time-domain analysis.”
