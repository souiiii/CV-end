# Camera Geometry and Calibration

These notes cover **C1–C4**: calibration fundamentals, affine transformations, vanishing points/lines and their uses, and camera-parameter estimation. The primary source is [CameraGeometry_CV_IIIT.pptx](../Sources/CameraGeometry_CV_IIIT.pptx): transformations in slides 10–38, projection in 39–49, vanishing geometry in 54–60, calibration in 61–80, and height measurement in 86–96.

**PYQ key:** P24 denotes `CV EndSem May 24.pdf`; PIP denotes `Image_Processing_Question_Papers_Extracted.pdf`. Page numbers are 1-based PDF pages. Numerical examples here are constructed examples.

**Notation throughout:** Vectors are columns. A world point is $\mathbf X_w=(X_w,Y_w,Z_w)^T$; its camera coordinates are $\mathbf X_c=(X_c,Y_c,Z_c)^T$. A pixel is $\mathbf p=(u,v)^T$, with homogeneous form $\tilde{\mathbf p}=(u,v,1)^T$. $R$ maps world directions to camera directions, $\mathbf t$ is the **world-to-camera translation**, and $\mathbf C_w$ is the camera centre in world coordinates. These last two quantities are different.

## 1. Camera Calibration: Meaning and Purpose

### What calibration estimates

Camera calibration estimates the parameters that relate known 3D scene points to their measured image pixels. It determines the camera's internal imaging geometry and its position/orientation relative to a chosen world frame. Real-camera calibration may also estimate lens-distortion coefficients.

It is useful for accurate measurements, distortion correction, pose estimation and geometric interpretation of images. Calibration alone does **not** recover a unique 3D point from a single pixel: that pixel usually identifies a viewing ray, and additional geometric information is needed to determine depth.

```text
World point -- rotation and translation --> camera coordinates
            -- perspective division ----> normalised image coordinates
            -- camera intrinsics --------> pixel coordinates
```

### Intrinsic and extrinsic parameters

**Intrinsics** describe the conversion from normalised image coordinates to pixels. A general intrinsic matrix is

$$
K=\begin{bmatrix}
f_x&\gamma&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
$$

- $f_x,f_y$ are focal lengths measured in pixels along the two image axes. If the physical focal length is $f$ and pixel pitches are $s_x,s_y$ in length units per pixel, then $f_x=f/s_x$ and $f_y=f/s_y$.
- $(c_x,c_y)$ is the **principal point**, where the optical axis meets the image plane, expressed in pixel coordinates. It need not be exactly the centre of the stored image.
- $\gamma$ is skew. It is usually taken as zero for perpendicular pixel axes.

**Extrinsics** describe the rigid change of frame:

$$
\mathbf X_c=R\mathbf X_w+\mathbf t,
\qquad R^TR=I,\quad\det R=+1.
$$

Rotation has three degrees of freedom and translation has three, giving six extrinsic degrees of freedom. The nine entries of $R$ are constrained and are not nine independent parameters.

The general $K$ above has five intrinsic degrees of freedom after setting $K_{33}=1$; imposing zero skew leaves four. The general projective camera matrix has 11 degrees of freedom. A zero-skew physical camera model imposes an additional constraint, so these counts must not be mixed indiscriminately.

### Camera centre versus translation

The camera centre maps to the camera-frame origin:

$$
\mathbf0=R\mathbf C_w+\mathbf t
\quad\Rightarrow\quad
\boxed{\mathbf C_w=-R^T\mathbf t}.
$$

Equivalently, $\mathbf X_c=R(\mathbf X_w-\mathbf C_w)$. The lecture sometimes writes this latter form with a symbol resembling translation. These notes consistently reserve $\mathbf t$ for the additive translation in $R\mathbf X_w+\mathbf t$.

**5-mark question:** Define camera calibration, distinguish intrinsic and extrinsic parameters, and explain why one image point does not uniquely determine depth.

## 2. Pinhole Projection and the Camera Matrix

### What the pinhole model describes

The ideal pinhole model projects scene points along rays through a single centre of projection. Use a virtual image plane in front of the centre to obtain positive focal-length equations. A physical screen behind the pinhole gives an inverted image and corresponding sign changes.

For an image plane at $Z_c=f$, similar triangles give

$$
\frac{x_i}{f}=\frac{X_c}{Z_c},\qquad
\frac{y_i}{f}=\frac{Y_c}{Z_c},
$$

so

$$
x_i=f\frac{X_c}{Z_c},\qquad y_i=f\frac{Y_c}{Z_c}.
$$

Here $(x_i,y_i)$ are metric image-plane coordinates. The division by depth is the source of perspective size change. An object of fixed height has projected height proportional to $f/Z_c$; it becomes smaller as distance increases.

Define normalised coordinates $x_n=X_c/Z_c$, $y_n=Y_c/Z_c$. In pixels,

$$
u=f_xx_n+\gamma y_n+c_x,\qquad
v=f_yy_n+c_y.
$$

### Homogeneous projection

Combining intrinsics and extrinsics gives

$$
\boxed{s\tilde{\mathbf p}
=K[R\mid\mathbf t]\tilde{\mathbf X}_w
=P\tilde{\mathbf X}_w},
$$

where $\tilde{\mathbf X}_w=(X_w,Y_w,Z_w,1)^T$, $P$ is $3\times4$, and the scale is $s=Z_c$ for a consistently normalised $K[R\mid\mathbf t]$.

If $P\tilde{\mathbf X}_w=(a,b,c)^T$, the pixel is $(a/c,b/c)$, provided $c\ne0$. Multiplying all entries of $P$ by a nonzero scalar does not change the projected pixel; hence its 12 entries represent only 11 independent projective parameters.

This convention agrees with the [OpenCV pinhole-camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). It also explains why image-plane signs must be changed consistently rather than copied between different conventions.

### Worked projection and parameter calculation

Let $R=I$, $\mathbf t=\mathbf0$, $f_x=f_y=800$, $\gamma=0$, and $(c_x,c_y)=(320,240)$. A point $\mathbf X_w=(1,2,10)^T$ projects to

$$
u=800\frac1{10}+320=400,\qquad
v=800\frac2{10}+240=400.
$$

The focal lengths are in pixels; the 3D coordinates may be in metres as long as all 3D lengths use the same unit.

For a simple parameter-estimation example, suppose $R=I$, $\mathbf t=0$, and two known ratios $X_c/Z_c$ are 0.1 and 0.2, observed at $u=400$ and $u=480$. Then

$$
400=0.1f_x+c_x,\qquad480=0.2f_x+c_x.
$$

Subtracting gives $80=0.1f_x$, so $f_x=800$ and $c_x=400-80=320$. General calibration estimates many parameters simultaneously from many correspondences.

### Projection versus affine transformation

Perspective projection contains depth-dependent division and generally does not preserve parallelism, lengths or angles. It is not a 2D affine transformation of an arbitrary 3D scene. Orthographic projection drops depth, and weak perspective approximates a small depth range by a common scale. Neither justifies claiming that arbitrary 3D lengths, areas and angles survive projection unchanged.

**5-mark question:** Derive the pinhole equations using similar triangles, explain $P=K[R\mid\mathbf t]$, and solve a small projection problem.

## 3. Affine Transformations

### What an affine transformation is

An affine transformation combines a linear transformation with translation:

$$
\begin{bmatrix}x'\\y'\end{bmatrix}
=A\begin{bmatrix}x\\y\end{bmatrix}+\mathbf d,
\qquad
A=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{bmatrix}.
$$

For an invertible affine transformation, $\det A\ne0$. Otherwise, a plane may collapse to a line or point.

Homogeneous coordinates allow translation to be included in matrix multiplication:

$$
\begin{bmatrix}x'\\y'\\1\end{bmatrix}
=\begin{bmatrix}
a_{11}&a_{12}&d_x\\
a_{21}&a_{22}&d_y\\
0&0&1
\end{bmatrix}
\begin{bmatrix}x\\y\\1\end{bmatrix}.
$$

There are **six degrees of freedom**: four entries of $A$ and two translations.

### Basic 2D matrices

The rotation below assumes Cartesian axes with $y$ increasing upward and positive angles anticlockwise.

$$
T(d_x,d_y)=\begin{bmatrix}1&0&d_x\\0&1&d_y\\0&0&1\end{bmatrix},
\qquad
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta&0\\\sin\theta&\cos\theta&0\\0&0&1\end{bmatrix},
$$

$$
S(s_x,s_y)=\begin{bmatrix}s_x&0&0\\0&s_y&0\\0&0&1\end{bmatrix},
\qquad
H_x(k)=\begin{bmatrix}1&k&0\\0&1&0\\0&0&1\end{bmatrix}.
$$

Translation shifts all points equally. Scaling stretches/compresses coordinate directions. The shear $H_x(k)$ gives $x'=x+ky$, $y'=y$. Reflection across the $x$-axis is $S(1,-1)$; reflection across the $y$-axis is $S(-1,1)$.

### Composition and order: worked example

For column vectors, the rightmost transformation acts first. Translating and then rotating gives $R T\tilde{\mathbf x}$. Reversing the operations gives $T R\tilde{\mathbf x}$; they generally differ.

Take $(x,y)=(1,0)$, translation $(2,0)$, and rotation $90^\circ$:

$$
(1,0)\xrightarrow{T}(3,0)\xrightarrow{R}(0,3),
$$

$$
(1,0)\xrightarrow{R}(0,1)\xrightarrow{T}(2,1).
$$

Thus $RT\ne TR$. To rotate about a pivot $(a,b)$ instead of the origin, use

$$
T(a,b)R(\theta)T(-a,-b).
$$

For example, rotating $(2,1)$ by $90^\circ$ about $(1,1)$ gives: subtract the pivot to get $(1,0)$, rotate to $(0,1)$, and add the pivot back to obtain $(1,2)$.

### Preserved properties

| Property | Invertible affine transform | Rigid transform: rotation and translation |
| --- | --- | --- |
| Straightness/collinearity | Preserved | Preserved |
| Parallelism | Preserved | Preserved |
| Ratios along the same line | Preserved | Preserved |
| Lengths and angles | Generally changed | Preserved |
| Area | Multiplied by $\lvert\det A\rvert$ | Preserved |

To justify the ratio property, let $\mathbf x_2=\mathbf x_1+\lambda\mathbf d$ and $\mathbf x_3=\mathbf x_1+\mu\mathbf d$. Their transformed differences are $\lambda A\mathbf d$ and $\mu A\mathbf d$, so their directed ratio remains $\lambda/\mu$. This does not imply preservation of arbitrary ratios between differently oriented lengths.

All planar areas under the same affine map scale by the same factor, so ratios of such areas are preserved. For $A=\operatorname{diag}(2,3)$, area scales by $|\det A|=6$, although lengths scale differently along different directions.

### 3D affine and rigid forms

In 3D,

$$
\begin{bmatrix}\mathbf X'\\1\end{bmatrix}
=\begin{bmatrix}A_{3\times3}&\mathbf d\\\mathbf0^T&1\end{bmatrix}
\begin{bmatrix}\mathbf X\\1\end{bmatrix}.
$$

A general 3D affine transform has 12 degrees of freedom. Its rigid special case uses $A=R$, with $R^TR=I$ and $\det R=1$, and has six degrees of freedom. An orthonormal matrix with determinant $-1$ includes a reflection and is not a proper rotation. Rotations about different 3D axes generally do not commute.

**5-mark question:** Define an affine transformation, give its homogeneous matrix and degrees of freedom, explain its invariants, and demonstrate composition order. **Relevant PYQ:** [PIP](../Papers/Image_Processing_Question_Papers_Extracted.pdf), p. 3 Q1(b), restricted to affine geometric transformations.

## 4. Vanishing Points and Vanishing Lines

### What a vanishing point is

A vanishing point is the image of an infinitely distant point in a particular 3D direction. Parallel scene lines have the same direction, so their projected images converge to the same vanishing point, except when the vanishing point is at infinity.

```text
Image of parallel scene lines:

   \        /
    \      /
     \    /
      \  /
       v       <- common vanishing point
```

### Deriving the vanishing-point equation

In camera coordinates, write a line as

$$
\mathbf X_c(\lambda)=\mathbf X_0+\lambda\mathbf d,
\qquad\mathbf d=(d_x,d_y,d_z)^T.
$$

The metric projected $x$ coordinate is

$$
x_i(\lambda)=f\frac{X_0+\lambda d_x}{Z_0+\lambda d_z}.
$$

Dividing numerator and denominator by $\lambda$ and taking $\lambda\to\infty$ gives

$$
x_v=f\frac{d_x}{d_z},\qquad y_v=f\frac{d_y}{d_z},\qquad d_z\ne0.
$$

The initial point disappears, so the result depends on direction rather than the line's position. In homogeneous pixels,

$$
\boxed{\tilde{\mathbf v}\sim K\mathbf d_c\sim KR\mathbf d_w}.
$$

Here $\sim$ means equality up to nonzero scale. Translation does not occur because a direction at infinity has homogeneous fourth coordinate zero.

If $d_z=0$, the lines are parallel to the image plane. Their vanishing point has homogeneous third coordinate zero and cannot be expressed as a finite pixel. Their images remain parallel; it is more precise to say “vanishing point at infinity” than “no vanishing point.”

**Example:** With $K=\operatorname{diag}(800,800,1)$ except for principal point $(320,240)$, and camera-frame direction $(1,0,2)$,

$$
u_v=800(1/2)+320=720,\qquad v_v=240.
$$

Every scene line with that direction has this same vanishing point. The direction $(1,0,0)$ instead produces a point at infinity.

### Vanishing line and horizon

A plane contains many line directions. The vanishing points of all directions within the plane lie on one **vanishing line**. For the ground plane, this line is the **horizon**. It is determined by the plane's orientation, and can lie outside the visible image.

A plane parallel to the image plane has its vanishing line at infinity, since all its line directions remain parallel to the image plane.

Two distinct vanishing points from two different directions in the same plane determine its vanishing line. Do not join two vanishing points belonging to unrelated planes and automatically call the result the ground horizon.

For camera-frame plane normal $\mathbf n$, a direction in the plane satisfies $\mathbf n^T\mathbf d=0$. Since $\mathbf d\sim K^{-1}\tilde{\mathbf v}$,

$$
\mathbf n^TK^{-1}\tilde{\mathbf v}=0.
$$

Therefore its vanishing line has homogeneous coordinates $\boldsymbol\ell_\infty\sim K^{-T}\mathbf n$. This shows why coplanar vanishing points are collinear.

### Construction from an image

For homogeneous points, $\boldsymbol\ell=\tilde{\mathbf p}_1\times\tilde{\mathbf p}_2$ represents the line through them. For homogeneous lines, $\tilde{\mathbf p}=\boldsymbol\ell_1\times\boldsymbol\ell_2$ represents their intersection. Divide by the third coordinate only when it is nonzero.

1. Identify two image lines corresponding to one family of parallel scene lines; intersect them to estimate a vanishing point.
2. Repeat for a second direction in the same plane.
3. Join the two vanishing points to obtain the plane's vanishing line.

More lines improve robustness to noisy endpoint measurements. Noisy or nearly parallel image lines can make finite vanishing-point estimates unstable.

**5-mark question:** Derive the vanishing point of a 3D direction, explain the infinity case, and distinguish a vanishing point, vanishing line and horizon.

## 5. Uses of Vanishing Geometry

### 5.1 Principal-point estimation from orthogonal directions

**What it does:** Under appropriate camera assumptions, three mutually perpendicular scene directions determine a triangle of vanishing points whose orthocentre is the principal point.

Assume zero skew, equal pixel focal lengths $f_x=f_y=f$, negligible/corrected distortion, and three **finite, non-collinear** vanishing points $\mathbf v_1,\mathbf v_2,\mathbf v_3$. Let the principal point be $\mathbf c=(c_x,c_y)^T$.

The corresponding camera directions are proportional to

$$
\begin{bmatrix}u_i-c_x\\v_i-c_y\\f\end{bmatrix}.
$$

Orthogonal directions have zero dot product, giving

$$
(\mathbf v_i-\mathbf c)^T(\mathbf v_j-\mathbf c)+f^2=0.
$$

Subtract the equations for pairs $(1,2)$ and $(1,3)$:

$$
(\mathbf v_1-\mathbf c)^T(\mathbf v_2-\mathbf v_3)=0.
$$

Thus the line from vertex $\mathbf v_1$ to $\mathbf c$ is perpendicular to the opposite side. Applying the same reasoning to another vertex identifies $\mathbf c$ as the intersection of altitudes, the **orthocentre**. It is not necessarily the centroid or image-array centre. Once $\mathbf c$ is known,

$$
f^2=-(\mathbf v_i-\mathbf c)^T(\mathbf v_j-\mathbf c).
$$

For example, $\mathbf v_1=(100,0)$, $\mathbf v_2=(-100,100)$ and $\mathbf v_3=(-100,-200)$ give orthocentre $(0,0)$. The first pair has dot product $-10{,}000$, so $f=100$ pixels. With unequal pixel scales or nonzero skew, the simple pixel-space orthocentre construction is not generally valid.

### 5.2 Height measurement using a reference

**What it does:** A known reference height, ground-plane horizon and vertical vanishing point allow an unknown height to be recovered from one perspective image. A raw ratio of image heights is unreliable when the objects stand at different depths.

Use the lecture's construction:

- $b,r$ are the reference base and top, with known real height $H_r$.
- $b_0,t_0$ are the target base and top, with unknown height $H$.
- Both bases lie on the same ground plane; the two objects are parallel to a common vertical direction with vanishing point $v_z$.

```text
Reference vertical             Target vertical
       r  (known height)              t0
       t  (transferred top)         /
       |                         /    line through q
       b-----------------------b0

q is where the image line b--b0 meets the ground-plane horizon.
The line q--t0 meets the reference vertical at t.
```

The construction is:

1. Find $v_z$ and the ground horizon $\boldsymbol\ell_\infty$ from parallel scene directions.
2. Find $q=(b\times b_0)\times\boldsymbol\ell_\infty$.
3. Find $t=(q\times t_0)\times(b\times r)$.

All points here are homogeneous when taking cross products. The ground direction $bb_0$ has vanishing point $q$. Therefore the world line represented by $qt_0$ is parallel to the base-to-base direction and transfers the target height to the reference vertical. The resulting world segment corresponding to $bt$ has length $H$.

### The necessary cross-ratio relation

The **cross-ratio** of four collinear points is invariant under perspective projection. For directed coordinates $a,b,c,d$ along a line, use

$$
\operatorname{CR}(A,B,C,D)=\frac{(c-a)(d-b)}{(d-a)(c-b)}.
$$

Choose the world points in the order $(B,V_\infty,T,R)$, with heights $(0,\infty,H,H_r)$. Taking the infinity limit gives

$$
\operatorname{CR}(B,V_\infty,T,R)=\frac{H}{H_r}.
$$

Consequently, on the common image vertical, using scalar directed coordinates $s_b,s_v,s_t,s_r$,

$$
\boxed{\frac{H}{H_r}
=\frac{(s_t-s_b)(s_r-s_v)}{(s_r-s_b)(s_t-s_v)}}.
$$

For the usual arrangement with $b,t,r$ on the same side of $v_z$, this can be written with positive image distances:

$$
\boxed{H=H_r\frac{|bt|\,|rv_z|}{|br|\,|tv_z|}}.
$$

The correction involving the vanishing point accounts for perspective. If $v_z$ is at infinity, the correction tends to 1 and $H/H_r=|bt|/|br|$ on the transferred common vertical.

**Numerical example:** After transfer, let $s_b=0$, $s_t=60$, $s_r=100$ and $s_v=300$ pixels, with $H_r=3$ m. Then

$$
H=3\frac{60(300-100)}{100(300-60)}
=3\frac{12{,}000}{24{,}000}=1.5\text{ m}.
$$

The direct ratio $3(60/100)=1.8$ m would ignore perspective and be wrong. Height measurement from vanishing geometry is also explained in the [MIT single-view-metrology chapter](https://visionbook.mit.edu/3d_scene_understanding_single_view.html).

**5-mark questions:** Explain the orthocentre method and its assumptions; or explain single-image height measurement with a horizon, reference height, construction diagram and cross-ratio relation.

## 6. Calibration from Known Correspondences

### What the practical procedure uses

A calibration object supplies known 3D locations whose corresponding 2D image features can be measured. The lecture uses a checkerboard cube as an example. A planar checkerboard can also be used across multiple views with an appropriate calibration method; it is not interchangeable with a general single-view 3D camera-matrix fit.

The common workflow is:

1. Choose a world frame and measure calibration-point coordinates.
2. Detect corresponding image points, spreading them across the image and useful depths/orientations.
3. Form projection constraints and estimate camera parameters.
4. Impose physical consistency, including a valid rotation and consistent parameter signs.
5. Reproject the known points, inspect the residuals and, where necessary, refine the model including distortion.

The ideal linear equations below assume negligible distortion or image coordinates corrected for distortion; significant unmodelled distortion biases the estimates.

A useful error measure is the root mean squared reprojection distance:

$$
e_{\mathrm{RMS}}=\sqrt{\frac1n\sum_{i=1}^{n}
\|\mathbf p_i-\hat{\mathbf p}_i\|_2^2},
$$

where $\mathbf p_i$ is the measured pixel and $\hat{\mathbf p}_i$ is the projected estimate. Its unit is pixels. Low training residual alone does not establish that calibration generalises; the observations must constrain the geometry adequately.

### 6.1 The lecture's Tsai procedure outline

**What it estimates:** The first taught procedure solves a constrained parameter representation, using a separately estimated principal point. Slide 75 describes an $n\times8$ homogeneous system for $n\ge8$ correspondences, with a rank-7 ideal case.

The reason an eight-component system arises can be seen from centred pixels $x_i=u_i-c_x$, $y_i=v_i-c_y$. With zero skew,

$$
x_i=f_x\frac{\mathbf r_1^T\mathbf X_i+t_x}{\mathbf r_3^T\mathbf X_i+t_z},\qquad
y_i=f_y\frac{\mathbf r_2^T\mathbf X_i+t_y}{\mathbf r_3^T\mathbf X_i+t_z},
$$

where $\mathbf r_j^T$ is row $j$ of $R$. Put $a=f_x/f_y$ and eliminate the common depth denominator:

$$
y_i a(\mathbf r_1^T\mathbf X_i+t_x)
-x_i(\mathbf r_2^T\mathbf X_i+t_y)=0.
$$

This is linear in the eight combined quantities

$$
\boldsymbol\beta=
(ar_{11},ar_{12},ar_{13},at_x,r_{21},r_{22},r_{23},t_y)^T.
$$

Each point contributes the row

$$
[y_iX_i,\ y_iY_i,\ y_iZ_i,\ y_i,\ -x_iX_i,\ -x_iY_i,\ -x_iZ_i,\ -x_i]
$$

to a system $A_T\boldsymbol\beta=0$. Solve up to scale using the smallest-eigenvalue eigenvector of $A_T^TA_T$, or the smallest right singular vector of $A_T$. Rotation constraints help recover parameter scale and rows of $R$; remaining projection equations supply focal/depth-translation information. Choose the physically consistent sign so calibration points are in front of the camera.

The ratio-equation explanation is consistent with [Horn's analysis of Tsai calibration](https://people.csail.mit.edu/bkph/articles/Tsai_Revisited_Old.pdf). The full Tsai algorithm has additional model/distortion details; they are not required by the lecture's outline. The **$n\ge8$ condition here belongs to this taught system**, not to every possible calibration algorithm.

### 6.2 Direct camera-matrix estimation: Faugeras–Toscani procedure

**What it estimates:** The second taught procedure first estimates the full $3\times4$ projective camera matrix, then decomposes it to obtain camera parameters. It is commonly expressed as a direct linear camera-matrix fit.

Let the rows of $P$ be $\mathbf p_1^T,\mathbf p_2^T,\mathbf p_3^T$, each of length 4. For a known homogeneous point $\mathbf q_i=\tilde{\mathbf X}_i$,

$$
u_i=\frac{\mathbf p_1^T\mathbf q_i}{\mathbf p_3^T\mathbf q_i},\qquad
v_i=\frac{\mathbf p_2^T\mathbf q_i}{\mathbf p_3^T\mathbf q_i}.
$$

Cross-multiplication removes the division:

$$
\mathbf p_1^T\mathbf q_i-u_i\mathbf p_3^T\mathbf q_i=0,
\qquad
\mathbf p_2^T\mathbf q_i-v_i\mathbf p_3^T\mathbf q_i=0.
$$

Stack the rows of $P$ into a 12-component vector $\mathbf m$. The two rows contributed by correspondence $i$ are

$$
\begin{bmatrix}
\mathbf q_i^T&\mathbf0^T&-u_i\mathbf q_i^T\\
\mathbf0^T&\mathbf q_i^T&-v_i\mathbf q_i^T
\end{bmatrix}\mathbf m=\mathbf0.
$$

For $n$ points, $A_D$ is $2n\times12$. Since camera scale is arbitrary, at least six correspondences in a **nondegenerate 3D configuration** are needed for this unconstrained fit. Counting equations alone does not prevent degeneracy. In particular, points all lying on one plane in one view do not determine a general $P$ uniquely.

The zero vector is a useless solution, so impose $\|\mathbf m\|_2=1$ and minimise

$$
\|A_D\mathbf m\|_2^2
=\mathbf m^TA_D^TA_D\mathbf m.
$$

The minimiser is the eigenvector for the smallest eigenvalue of $A_D^TA_D$, equivalently the right singular vector for the smallest singular value of $A_D$. Reshape it into $P$. In ideal nondegenerate noiseless data, the design matrix has rank 11; noise generally prevents an exact null solution. Normalising image/world coordinates before fitting improves numerical conditioning, and the resulting matrix must be transformed back afterward.

### 6.3 Recovering intrinsics, rotation and translation

Partition the estimated camera as

$$
P=[B\mid\mathbf p_4],\qquad B=KR,\quad\mathbf p_4=K\mathbf t.
$$

Use **RQ decomposition** of the left $3\times3$ block: its upper-triangular factor corresponds to $K$, and its orthogonal factor corresponds to $R$. Resolve sign/scale choices consistently, use $K_{33}=1$, take positive focal lengths in this convention, and enforce $\det R=+1$. Decomposition is not unique without such conventions.

After applying the same overall normalization to the camera blocks,

$$
\boxed{\mathbf t=K^{-1}\mathbf p_4},\qquad
\boxed{\mathbf C_w=-R^TK^{-1}\mathbf p_4=-B^{-1}\mathbf p_4}.
$$

The fourth column of $P$ is **not** directly the translation vector. If the camera is instead written $P=K[R\mid-R\mathbf C_w]$, recovering the centre requires the minus sign above. This resolves the sign ambiguity in lecture slide 80.

### Worked decomposition example

Suppose the normalised estimated matrix is

$$
P=\begin{bmatrix}
800&0&320&2400\\
0&800&240&2000\\
0&0&1&5
\end{bmatrix}.
$$

Here $B$ is already upper triangular, so a consistent decomposition is $K=B$, $R=I$. Solve $K\mathbf t=\mathbf p_4$ by back-substitution:

$$
t_z=5,\qquad
800t_y+240(5)=2000\Rightarrow t_y=1,
$$

$$
800t_x+320(5)=2400\Rightarrow t_x=1.
$$

Thus $\mathbf t=(1,1,5)^T$ and $\mathbf C_w=(-1,-1,-5)^T$. A world point at the origin projects to $(2400/5,2000/5)=(480,400)$, which agrees with direct projection of camera coordinates $(1,1,5)$.

### Comparing the taught procedures

| Feature | Tsai outline in the lecture | Camera-matrix procedure |
| --- | --- | --- |
| Initial unknown | Eight combined parameter quantities | Twelve entries of $P$, up to scale |
| Linear system | $n\times8$, with $n\ge8$ as taught | $2n\times12$, at least six nondegenerate 3D correspondences |
| Principal point | Supplied separately in the taught outline | Recovered through decomposition with the intrinsics |
| Parameter recovery | Uses physical constraints and remaining equations | Uses RQ of the left block, then translation recovery |

**5-mark questions:** Form the two camera-matrix equations from one correspondence; explain the homogeneous least-squares solution; or describe the taught calibration procedures and how RQ yields camera parameters. No supplied PYQ directly establishes a different calibration method as required.

## 7. Lens Distortion: Calibration Fundamentals

### What distortion means

Lens distortion is a systematic departure from the ideal pinhole mapping. It is different from the size change caused by perspective, and different from blur. Calibration estimates a model that permits correction of these positional errors.

**Radial distortion** depends on distance from the principal region and bends ideal straight-line projections. Common appearances are barrel distortion, where magnification decreases towards the edges, and pincushion distortion, where it increases. **Tangential/decentring distortion** arises from lens misalignment and displaces points asymmetrically.

For ideal normalised coordinates $(x_n,y_n)$, let $r^2=x_n^2+y_n^2$. A simple radial model is

$$
x_d=x_n(1+k_1r^2+k_2r^4),\qquad
y_d=y_n(1+k_1r^2+k_2r^4).
$$

$k_1,k_2$ are radial coefficients and $(x_d,y_d)$ are distorted normalised coordinates. Zero coefficients recover the ideal model. Tangential distortion uses separate coefficients, conventionally $p_1,p_2$; full higher-order distortion equations are beyond the required fundamentals. This distinction and model are verified in [OpenCV's calibration tutorial](https://docs.opencv.org/4.10.0/dc/dbb/tutorial_py_calibration.html).

Correction maps between distorted observations and the ideal camera model; it does not generally amount to subtracting one fixed offset from every pixel. **Relevant PYQ:** [May 2024 CSE CV paper](../Papers/CV%20EndSem%20May%2024.pdf), p. 1 Q6(e), asks for lens distortion and its types.

**5-mark question:** Explain lens distortion, distinguish radial and tangential effects, give a simple radial model and describe why calibration is needed.

## 8. Final Revision Checks

- State whether a point is in world, camera, normalised-image or pixel coordinates. Use a consistent virtual-image-plane convention.
- Distinguish the camera centre $\mathbf C_w$ from additive translation $\mathbf t$; remember $\mathbf C_w=-R^T\mathbf t$.
- For affine questions, give the homogeneous form, composition order and precise invariants. Perspective division is not affine.
- For vanishing geometry, connect **direction → point**, **plane directions → vanishing line**, and **ground plane → horizon**. Include the infinity case and applications.
- For estimation, explain correspondences, design-matrix dimensions, scale ambiguity, a nondegenerate point configuration, least-squares solution and parameter recovery.

Stereo reconstruction, epipolar-matrix derivations and structure from motion are outside these notes. Calibration applications are mentioned only to explain its purpose.
