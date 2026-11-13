# Object Detection and Tracking

These notes cover **D1–D2**, including the teacher's explicit emphasis on **convolutional approaches**. The supplied [CNN lecture](../Sources/CNN_CV_IIIT.pptx), especially slides 2–19 and 45–54, explains learned features but does not develop a complete detector or tracker. The [syllabus](../Syllabus/CV-7th-Semester-IT.pdf), p. 2, Module II, names tracking. The workflows below are bounded supplementary explanations, not a claim that particular named architectures were taught.

**PYQ key:** P24 denotes `CV EndSem May 24.pdf`, the one-page May 2024 CSE Computer Vision paper. Numerical examples here are illustrative, not quotations of PYQ data.

## 1. Detection, Classification and Tracking

### What the tasks mean

**Object detection** identifies object instances and their locations in an image. A typical output is a list of bounding boxes, class labels and confidence scores. **Object tracking** follows the same object over successive frames, maintaining its location and identity even when its appearance changes.

| Task | Question answered | Example output |
| --- | --- | --- |
| Image classification | What category describes this image? | “Person.” |
| Object detection | What objects occur, and where? | Two person boxes and one bicycle box. |
| Object tracking | Which observation belongs to which continuing object? | Person ID 7 at a new position in each frame. |

A detector can label a person correctly in every frame while still failing to preserve that person's identity. Conversely, a tracker initialised with a target box may follow the target without determining its semantic class.

For a continuous-coordinate bounding box, use

$$
\mathbf b=(x_1,y_1,x_2,y_2),\qquad x_2>x_1,\ y_2>y_1.
$$

Its width is $w=x_2-x_1$ and height is $h=y_2-y_1$. Another common representation is centre, width and height: $(x_c,y_c,w,h)$, with $x_c=(x_1+x_2)/2$ and $y_c=(y_1+y_2)/2$. State which convention a numerical problem uses.

**5-mark question:** Distinguish classification, detection and tracking using their outputs and a video example.

## 2. A Convolutional Object Detector

### What makes it convolutional

A CNN detector learns image features with convolutional layers and uses those features to predict both **what** an object is and **where** it is. A kernel applies shared learned weights to local image regions. Early features often respond to edges or textures; later features combine information over larger receptive fields into more task-specific patterns.

The essential detector adds localisation predictions to convolutional feature extraction. A classifier that ends with one label for the whole image does not automatically become a detector.

### Working pipeline

```text
Input image -> preprocessing -> CNN feature maps
                                  |
                   +--------------+--------------+
                   |                             |
            class/object scores           box predictions
                   +--------------+--------------+
                                  |
                    decode boxes and filter scores
                                  |
                  remove duplicate overlapping boxes
                                  |
                       final labelled detections
```

1. **Preprocess the image.** Resize and normalise inputs consistently. Predicted coordinates must eventually be mapped back to the original image size.
2. **Extract features.** Convolutional layers compute feature maps over the image. Shared features avoid independently learning a separate classifier for every possible location.
3. **Predict candidates.** At candidate regions or feature-map locations, predict category/object scores and box coordinates or offsets. Large and small objects occupy different numbers of feature cells, which makes scale variation challenging.
4. **Decode and select.** Convert predictions into image-coordinate boxes, reject low-confidence candidates and suppress redundant boxes.

### Two common workflow patterns

**Two-stage detection** first proposes likely object regions, then classifies and refines them. In the Faster R-CNN example, a region-proposal network shares convolutional features with the detection network. This illustrates the proposal-then-classification workflow; detailed layer tables are not needed. [Original Faster R-CNN paper](https://arxiv.org/abs/1506.01497).

**One-stage detection** directly predicts boxes and class information from the image's feature representation without a separate proposal-classification stage. The original YOLO formulation treats detection as prediction of boxes and associated class probabilities in one network evaluation. This illustrates the direct-prediction workflow, not a requirement to memorise YOLO versions. [Original YOLO paper](https://arxiv.org/abs/1506.02640).

| Pattern | Main idea | Practical consideration |
| --- | --- | --- |
| Two-stage | Select candidate regions, then classify/refine them. | Adds a region-processing stage. |
| One-stage | Predict detection outputs directly from feature locations. | Usually offers a simpler inference pipeline. |

Do not state that one pattern is universally more accurate: results depend on the model, image scale, data and computation.

### How detection is learned

Training requires annotated images with object categories and ground-truth boxes. A generic loss combines classification and localisation terms:

$$
L=L_{\mathrm{cls}}+\lambda L_{\mathrm{loc}},\qquad \lambda>0.
$$

$L_{\mathrm{cls}}$ penalises incorrect category/background decisions. $L_{\mathrm{loc}}$ penalises inaccurate boxes for matched object examples; background examples have no object box to regress. The factor $\lambda$ balances the two terms. The exact assignment rule and loss depend on the detector, so a universal architecture-specific formula should not be assumed.

For a simple illustration, if the predicted centre is $(52,41)$ and the target is $(50,40)$, its centre error is $(2,1)$. Reducing classification error alone cannot eliminate that localisation error. Both prediction tasks must be trained.

**5-mark question:** Explain a CNN detection pipeline with a diagram, showing how classification and localisation differ. A complete answer should include candidate prediction and duplicate removal.

## 3. Box Overlap and Non-Maximum Suppression

### Intersection over Union

**Intersection over Union (IoU)** measures the overlap between two boxes:

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}
=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

Here $|\cdot|$ denotes area. IoU lies between 0 and 1: disjoint boxes have 0, and identical nonzero-area boxes have 1.

For boxes in corner format, intersection width is

$$
w_I=\max\!\left(0,\min(x_{2A},x_{2B})-\max(x_{1A},x_{1B})\right),
$$

with an analogous expression for $h_I$. Then $|A\cap B|=w_Ih_I$. The outer maximum handles disjoint boxes.

### Worked IoU calculation

Let $A=(0,0,4,4)$ and $B=(1,1,5,5)$. Each has area 16. Their intersection extends from $(1,1)$ to $(4,4)$, giving area $3\cdot3=9$. Thus

$$
\operatorname{IoU}(A,B)=\frac9{16+16-9}=\frac9{23}\approx0.391.
$$

### Non-maximum suppression (NMS)

NMS removes redundant predictions around the same object:

1. Sort candidate boxes by score.
2. Keep the highest-scoring remaining box.
3. Suppress lower-scoring boxes whose IoU with it exceeds a chosen threshold.
4. Repeat on the remaining candidates.

This is the standard greedy operation described in the [Torchvision NMS documentation](https://docs.pytorch.org/vision/main/generated/torchvision.ops.nms.html). In a class-aware detector, apply it separately for each class so that different categories are not suppressed merely because they overlap.

If $A$ above has score 0.90 and $B$ has score 0.75, an NMS threshold of 0.30 keeps $A$ and suppresses $B$ because $0.391>0.30$. With threshold 0.50, neither suppresses the other. A third disjoint box survives either threshold if its score passes the confidence cutoff.

IoU is geometric agreement; a confidence score is a model prediction. They are different quantities. A low NMS threshold can mistakenly suppress different nearby objects, while a high threshold may leave duplicates.

**5-mark question:** Define IoU, calculate it for two boxes, and explain NMS with scores and a threshold.

## 4. Object Tracking Across Frames

### What tracking maintains

A track is a continuing hypothesis about one physical object. It typically stores an ID, recent location, appearance information and a status such as active, temporarily missing or terminated. A trajectory is the sequence of its estimated locations over time.

```text
Current frame -> detections/candidate regions -> CNN appearance features
                                                   |
Previous tracks -> expected locations --------> association
                                                   |
                         update matched tracks / initialise new tracks
                         retain or terminate unmatched tracks
                                                   |
                                  labelled tracks for this frame
```

### Main steps

**Initialisation:** Start a target from a given box or a reliable detection, and assign an ID. Extract a template or feature vector that describes its appearance.

**Search or detection:** In the next frame, obtain candidate locations. A single-target tracker may search near the previous location. A tracking-by-detection system uses a detector to supply observations in each frame.

**Association:** Decide which candidate corresponds to which existing track, using appearance similarity and spatial consistency. Reject implausible matches instead of forcing an assignment.

**Update:** Update location and, when the match is reliable, appearance information. Unmatched detections can begin new tracks. Unmatched tracks may remain temporarily missing; repeated absence leads to termination.

### Simple motion consistency

If a target centre moved from $\mathbf c_{t-2}$ to $\mathbf c_{t-1}$ over equal frame intervals, a simple prediction is

$$
\hat{\mathbf c}_t=\mathbf c_{t-1}+(\mathbf c_{t-1}-\mathbf c_{t-2}).
$$

For centres $(20,30)$ and $(24,31)$, the displacement is $(4,1)$ pixels per frame and the next predicted centre is $(28,32)$. This is a simple consistency estimate, not a guarantee: abrupt motion can invalidate it. No full motion-filter derivation is needed for this scope.

### Main failure modes

| Challenge | Why it matters | Sensible response |
| --- | --- | --- |
| Occlusion | The object becomes partly or fully invisible. | Retain a missing track briefly; avoid updating its appearance with the occluder. |
| Similar-looking objects | Appearance alone may confuse identities. | Combine appearance with location and one-to-one matching. |
| Scale, pose or illumination change | The current appearance differs from the initial template. | Use robust learned features and cautious updates. |
| Drift | An incorrect update makes the model increasingly describe the background or another object. | Gate low-confidence matches and refresh from reliable detections. |
| Missed/false detections | Observations disappear or appear spuriously. | Use track confirmation and controlled start/termination rules. |

**5-mark question:** Explain the tracking cycle and discuss occlusion and drift. Include initialisation, association and update rather than merely listing challenges.

## 5. Convolutional Features for Tracking and Re-Identification

### What the CNN contributes

A CNN can map an object crop to an **appearance embedding**, a vector whose distances reflect visual/identity similarity. It is trained so that observations of the same identity are compatible despite viewpoint or lighting changes. The tracker uses these vectors to compare a previous target with current candidates.

The value of learned appearance association is supported by the [Deep SORT paper](https://arxiv.org/abs/1703.07402), which combines visual appearance with tracking to reduce identity switches. The generic example below explains this principle; the full named algorithm is not an additional chapter.

### Feature matching

Let $\phi(\mathcal R)$ be the CNN embedding for region $\mathcal R$. Cosine similarity is

$$
s(\mathbf a,\mathbf b)=\frac{\mathbf a^T\mathbf b}{\|\mathbf a\|\|\mathbf b\|}.
$$

For unit-normalised embeddings it is simply the dot product. A corresponding distance is $d_{\mathrm{app}}=1-s$, so smaller values mean greater similarity.

If a stored target feature is $(1,0)$ and a unit candidate feature is $(0.8,0.6)$, then $s=0.8$ and $d_{\mathrm{app}}=0.2$. A candidate with feature $(0,1)$ gives $s=0$, so the first candidate is a better appearance match. A plausible location is still needed: similarity alone is not proof of identity.

For a single target, a convolutional tracking approach can extract features for the previous template and candidate regions, select the most similar plausible region, and update only when confidence is adequate.

For multiple objects, combine appearance with a location cost, for example

$$
C_{ij}=\lambda d_{\mathrm{app}}(i,j)
+(1-\lambda)\left[1-\operatorname{IoU}(\hat{\mathbf b}_i,\mathbf b_j)\right],
\qquad0\le\lambda\le1.
$$

Here $i$ indexes an existing track, $j$ a detection, $\hat{\mathbf b}_i$ is its expected box, and $\lambda$ sets the weight on appearance. This is an illustrative association cost, not a universal tracking formula. Reject pairs beyond motion/appearance gates, then seek a one-to-one assignment with low total cost.

For an illustrative cost matrix

$$
\begin{array}{c|cc}
&D_1&D_2\\\hline
T_1&0.10&0.80\\
T_2&0.70&0.20
\end{array},
$$

the assignment $T_1\to D_1$, $T_2\to D_2$ costs $0.30$, versus $1.50$ for the crossed assignment. Both tracks must not claim the same detection.

### Person re-identification

Person re-identification matches observations of the same person across separate times or cameras. A concise CNN approach is: crop detected people, extract normalised embeddings, compare with stored identities, and rank or threshold the distances. Training uses identity-labelled examples; inference compares embeddings.

Re-identification can help a tracker recover an identity after an interruption, but it does not by itself estimate a continuous trajectory. **Relevant PYQ:** [May 2024 CSE CV paper](../Papers/CV%20EndSem%20May%2024.pdf), p. 1 Q6(b), asks about CNNs in person re-identification.

**5-mark question:** Explain CNN-based appearance matching for tracking or person re-identification, including feature extraction, similarity, association and a limitation.

## 6. Related Application: CNN Background/Foreground Detection

### What it means

Background/foreground detection separates background pixels from pixels belonging to foreground activity. It supplies a **pixel mask**, whereas object detection supplies object instances such as labelled boxes. This distinction matters for **P24 p. 1 Q6(d)**, which asks about CNNs in background detection.

A bounded CNN explanation is to input a frame, optionally together with a reference background, compute convolutional features, and produce a foreground probability $p_{\mathrm{fg}}[x,y]$ at each pixel. Thresholding produces a foreground mask; the complementary binary mask identifies background. Supervised training uses foreground/background-labelled pixels.

The network can learn appearance cues beyond a raw intensity difference. Nevertheless, shadows, camera motion and unfamiliar scenes can cause errors. A mask has neither object IDs nor necessarily semantic classes, so detection/association steps are still required if the goal is tracking. This application does not require a separate segmentation-architecture survey.

## 7. Exam Preparation and Boundaries

For an answer on **detection**, include the input/output, CNN features, class and box predictions, and selection of final boxes. For **tracking**, include temporal identity, candidate generation, association, updating and failure handling. For **convolutional approaches**, explain what the learned features represent and how they feed the relevant decision.

| Verified PYQ reference | How to use it |
| --- | --- |
| P24 p. 1 Q6(b) | Practice the CNN person re-identification explanation in Section 5. |
| P24 p. 1 Q6(d) | Practice the short foreground/background application in Section 6. It is not a complete object-detector question. |
| P24 p. 1 Q6(a), KLT tracker | Conditional in the scope; it does not establish a required classical-tracker chapter. |

The exact taught detector/tracker family remains unknown. The notes cover the confirmed mechanisms with generic workflows; detailed R-CNN/YOLO variants, KLT, optical flow, Kalman/particle filters and named tracking architectures are not added as independent chapters. CNN layer formulas are developed in [CNN and Generative Models](04_CNN_and_Generative_Models.md), rather than repeated here.
