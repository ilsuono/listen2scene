---
abstract: |
  We propose a mesh-based neural network (MESH2IR) to generate acoustic
  impulse responses (IRs) for indoor 3D scenes represented using a mesh.
  The IRs are used to create a high-quality sound experience in
  interactive applications and audio processing. Our method can handle
  input triangular meshes with arbitrary topologies (2K - 3M triangles).
  We present a novel training technique to train MESH2IR using energy
  decay relief and highlight its benefits. We also show that training
  MESH2IR on IRs preprocessed using our proposed technique significantly
  improves the accuracy of IR generation. We reduce the non-linearity in
  the mesh space by transforming 3D scene meshes to latent space using a
  graph convolution network. Our MESH2IR is more than 200 times faster
  than a geometric acoustic algorithm on a CPU and can generate more
  than 10,000 IRs per second on an NVIDIA GeForce RTX 2080 Ti GPU for a
  given furnished indoor 3D scene. The acoustic metrics are used to
  characterize the acoustic environment. We show that the acoustic
  metrics of the IRs predicted from our MESH2IR match the ground truth
  with less than 10% error. We also highlight the benefits of MESH2IR on
  audio and speech processing applications such as speech
  dereverberation and speech separation. To the best of our knowledge,
  ours is the first neural-network-based approach to predict IRs from a
  given 3D scene mesh in real-time.
author:
- Anton Ratnarajah
- Zhenyu Tang
- Rohith Aralikatti
- Dinesh Manocha
bibliography:
- sample-base.bib
title: "MESH2IR: Neural Acoustic Impulse Response Generator for Complex
  3D Scenes"
---

::: CCSXML
\<ccs2012\> \<concept\>
\<concept_id\>10010147.10010341.10010349.10010359\</concept_id\>
\<concept_desc\>Computing methodologies Real-time
simulation\</concept_desc\>
\<concept_significance\>300\</concept_significance\> \</concept\>
\<concept\>
\<concept_id\>10010147.10010341.10010349.10010360\</concept_id\>
\<concept_desc\>Computing methodologies Interactive
simulation\</concept_desc\>
\<concept_significance\>300\</concept_significance\> \</concept\>
\<concept\> \<concept_id\>10010147.10010257\</concept_id\>
\<concept_desc\>Computing methodologies Machine
learning\</concept_desc\>
\<concept_significance\>300\</concept_significance\> \</concept\>
\<concept\>
\<concept_id\>10010147.10010371.10010372.10010374\</concept_id\>
\<concept_desc\>Computing methodologies Ray tracing\</concept_desc\>
\<concept_significance\>100\</concept_significance\> \</concept\>
\</ccs2012\>
:::

# INTRODUCTION

Rapid developments in interactive applications (e.g., games, virtual
environments, speech recognition, etc.) demand realistic sound effects
in complex dynamic indoor environments with multiple sound sources.
Generating realistic sound effects is still a challenging problem
because of the geometric and aural complexity of the real-world
environment. The geometric complexity is a measure of the number of
vertices and faces needed to represent the complex environment as a 3D
mesh. The aural complexity depends on the number of sound sources, the
number of static and dynamic objects in the environment and the acoustic
effects (e.g., early reflection, late reverberation, diffraction,
scattering, etc.). The way the sound propagates from a sound source to
the listener can be modeled as an impulse response
(IR) [@intro:soundprop]. We can generate sound effects by convolving the
IR with a dry sound. The IRs are used to generate plausible sound
effects in many interactive applications used for games and AR/VR: Steam
Audio [@steam-audio], Project Acoustics [@projectacoustic], and Oculus
Spatializer [@Oculus-spatializer].

Recently, many machine learning-based algorithms are proposed to
synthesize the sounds in musical instruments [@music1; @music2],
estimate the acoustic material properties, and model the acoustic
effects from finite objects [@finite4; @finite1; @finite2; @finite3].
Neural-network-based IR generators [@ir-gan; @fast-rir] for simple rooms
are proposed for speech processing applications (e.g., far-field speech
recognition, speech enhancement, speech separation, etc.).
FAST-RIR [@fast-rir] is a GAN-based IR generator that takes
shoe-box-shaped room dimensions, listener and source positions, and
reverberation time as inputs and generates a large IR dataset on the
fly. Gaming applications demand fast IR generators for complex scenes.
Complex indoor 3D scenes with furniture can be represented in detail
using mesh models. Traditional geometric IR simulators take a 3D scene
mesh-model and listener and source positions as inputs and generate
realistic environmental sound effects [@gsound]. The complexity of
geometric IR simulators increases exponentially with the reflection
depth [@image-method] and makes them impractical for interactive
applications. No current simulation and learning methods can compute
real-time IRs for unseen complex dynamic scenes.

Sound propagation can be accurately simulated using wave-based methods.
The wave-based methods solve wave equations using different numerical
solvers such as the boundary-element method [@boundary-element], the
finite-difference time-domain simulation [@FDTD], etc. The complexity of
the wave-based approach grows as the fourth power of the frequency and
is limited to static scenes. Geometric acoustic algorithms are a less
complex alternative to the wave-based method. Geometric acoustic
algorithms can handle complex environments with dynamic
objects [@dynamic1] and multiple sources [@source1]. However, geometric
acoustic methods do not model the low-frequency components of the IRs
accurately because of the ray assumption. The sound wave can be treated
as a ray when the wavelength of the sound is smaller than the obstacles
in the environment. At low frequencies under 500 Hz, the ray assumption
is not valid for most scenes and results in significant simulation
errors. Hybrid sound propagation algorithms [@precompute1; @GWA] combine
IRs from wave-based and geometric techniques to generate accurate IRs in
the human aural range for complex dynamic scenes. Generating IRs
corresponding to thousands of sound sources at an interactive rate is
not possible with the hybrid method because of its complexity.

**Main Contributions**: We propose a novel learning-based IR generator
(MESH2IR) to generate realistic IRs for furnished 3D scenes with
arbitrary topologies (i.e., meshes with 2000 faces to 3 million faces)
not seen during the training. For a given complex scene, MESH2IR can
generate IRs for any listener and source positions. We perform mesh
simplification to even handle complex 3D scenes represented using a mesh
with millions of triangles. Our mesh encoder network significantly
reduces the input data size by transforming the 3D scene meshes into
low-dimensional vectors of a latent space. Mesh simplification and its
encoder network allow us to handle general 3D scenes meshes with a
varying number of faces. We present an efficient approach to preprocess
the IR training dataset and show that training MESH2IR on preprocessed
dataset gives a significant improvement in the accuracy of IR
generation. We also evaluate the contribution of energy decay relief in
improving the IRs generated from MESH2IR. We train our MESH2IR on an IR
dataset computed using a hybrid sound propagation algorithm [@GWA].
Therefore, the predicted IRs from MESH2IR exhibit good accuracy for both
low-frequency and high-frequency components. MESH2IR can generate more
than 10,000 IRs for a given indoor 3D scene. We show that our MESH2IR
can generate IRs 200 times faster than a geometric acoustic
simulator [@pygsound] on a single CPU. We evaluate the accuracy of the
predicted IRs from MESH2IR using power spectrum and acoustic metrics
which characterize the acoustic environment. Our MESH2IR can predict the
IRs for unseen 3D scenes during training with less than 10% error in
three different acoustic metrics. We also show that far-field speech
augmented using the IRs generated from MESH2IR significantly improves
the performance in speech processing applications.

# RELATED WORKS

We first give an overview of IR simulators that can compute IRs at
interactive rates in Section [2.1](#IRsim){reference-type="ref"
reference="IRsim"}. In Section [2.2](#IRalgo){reference-type="ref"
reference="IRalgo"}, we summarize the deep learning-based algorithms
used to predict IRs and describe the prior learning-based IR generator.
We highlight various applications in audio and speech processing that
can use the predicted IRs in Section [2.3](#IRapp){reference-type="ref"
reference="IRapp"}. Finally, we mention different publicly available
indoor 3D scenes datasets and give an overview of prior IR datasets.

## PHYSICS-BASED IR COMPUTATION {#IRsim}

Many wave-based, geometric-based, and hybrid interactive IR simulation
algorithms have been proposed to simulate IRs for complex
scenes [@SURVEY_SOUND]. Wave-based and hybrid algorithms precompute the
IRs for a static scene and, at runtime, the IR for an arbitrary listener
position is calculated by efficient interpolation
techniques [@precompute1; @precompute2; @precompute3]. These
precomputation-based interactive IR simulation algorithms can be used
only for static scenes [@SURVEY_SOUND]. Geometric-based algorithms are
proposed for interactive IR simulation in dynamic
scenes [@dynamic1; @dynamic2; @dynamic3]. The limitations of
geometric-based techniques, such as simulation error at lower
frequencies, are inherent in these interactive geometric-based
algorithms. There are no general physics-based algorithms known for
computing accurate IRs, including low-frequency components, for general
dynamic scenes.

## LEARNING-BASED IR COMPUTATION {#IRalgo}

Recently, neural-network-based methods have been developed to estimate
IRs from the reverberant speech signal [@ir_estimate] or from a single
image of the environment [@imageRIR; @image2reverb], to translate
synthetic IRs to real-word IRs [@ts-rir], to interpolate IRs [@deepIR],
and to augment the number of IRs [@augment_IR; @ir-gan]. FAST-RIR is a
learning-based IR generator that is trained to generate specular and
diffuse reflections for a given empty shoe-box-shaped room. FAST-RIR may
not compute low-frequency components of IRs accurately.

## AUDIO PROCESSING USING IR {#IRapp}

IRs are used in a wide range of practical applications such as
audio-visual
navigation [@soundspaces; @navigation1; @neural_acoustic_field],
acoustic matching [@acoustic_matching], sound rendering [@source1],
sound source localization [@source_localize], speech
enhancement [@speech_enhancement], speech
recognition [@speech_recognition1; @speech_recognition2], and speech
separation [@speech_seperate2; @speech_seperate1]. In most applications,
learning-based networks are trained on large, diverse synthetic datasets
and tested in real-world environments [@pygsound]. The performance of
the deep neural network trained for a particular application is depended
on the similarity of the synthetic training dataset and the real-world
test environment [@ir-gan]. Synthetic training datasets are created
using IR generators. Therefore, the accuracy of the IR generator plays
an important role in practical applications that depends on IRs.

## INDOOR 3D SCENES & IR DATABASES {#3Dscene}

The indoor 3D scenes can be either captured using RGB-D videos and
reconstructed as meshes (real-world
scenes) [@acquired1; @acquired2; @acquired3] or created by humans using
professionally designed software (synthetic
models) [@3dfront; @designed1]. The mesh quality in the real-world scene
datasets is not as good as the quality of the synthetic models because
3D scene reconstruction with accurate geometric details is challenging
with existing computer vision methods and capturing hardware. Among the
synthetic model datasets, 3D-FRONT [@3dfront] contains large-scale
synthetic furnished indoor 3D scenes with fine geometric and texture
details. The 3D-FRONT dataset has 6813 CAD houses, where 18,968 rooms
are furnished with 13,151 3D furniture objects. The furniture is placed
in varying numbers in meaningful locations in each room (e.g., living
room, dining room, kitchen, bedroom, etc.)

The publicly available IR datasets are either recorded in a real-world
environments (recorded IR) [@realIR1; @realIR2; @realIR3; @realIR4] or
generated using IR simulation tools (synthetic
IR) [@syntheticIR1; @syntheticIR2; @soundspaces; @GWA]. The recorded IR
datasets are limited in size (fewer than 5000 IRs) and number of
environments (fewer than 10 rooms), and no sufficient information on the
recording conditions is provided to train a deep learning model.
Synthetic IR datasets like SoundSpaces [@soundspaces] and GWA [@GWA]
contain millions of IRs. The IRs in SoundSpaces is simulated using a
geometric acoustic method [@gsound] and GWA dataset contains
high-quality IRs computed using a hybrid method.

<figure id="full_architecture" data-latex-placement="h">
<embed src="full_archi123-eps-converted-to.pdf" />
<figcaption>The architecture of our MESH2IR. Our mesh encoder network
encodes a indoor 3D scene mesh to the latent space. The mesh latent
vector and the source and listener locations are combined to produce a
scene vector embedding (<span
class="math inline"><strong>π</strong><sub><strong>A</strong></sub></span>).
The generator network generates an IR corresponding to the input scene
vector embedding. For the given scene vector embedding, the
discriminator network discriminates between the generated IR and the
ground truth IR during training.</figcaption>
</figure>

# MESH2IR: OUR APPROACH {#approach}

<figure id="graph_network" data-latex-placement="h">
<embed src="mesh123-eps-converted-to.pdf" />
<figcaption>The expansion of our mesh encoder in Figure <a
href="#full_architecture" data-reference-type="ref"
data-reference="full_architecture">1</a>. Our encoder network transforms
the indoor 3D scene mesh into a latent vector. The topology information
(edge connectivity) and the node features (vertex coordinates) are
extracted from the mesh and passed to our graph neural
network.</figcaption>
</figure>

## OVERVIEW

Our goal is to predict the IR for the given indoor 3D scene and the
source and listener positions
(Equation [\[overall_network\]](#overall_network){reference-type="ref"
reference="overall_network"}). The 3D scenes are represented as
triangular meshes and we do not make any assumptions about the topology
of the 3D scenes. The mesh format describes the shape of an object using
vertices, edges, and faces consisting of triangles represented using 3D
Cartesian coordinates (x,y,z). The source positions and listener
positions are also represented using 3D Cartesian coordinates. For each
scene material (e.g., furniture, floor, wall, ceiling, etc.), we do not
explicitly control the characteristics of the scene materials (the
amount of absorption or scattering of sound by the scene material). The
IRs used to train our MESH2IR is computed by considering the
characteristic of scene material in every indoor 3D scene. Therefore,
our trained MESH2IR randomly assigns the characteristics of scene
materials based on their shape.

The IR is the response of an impulse signal emitted in an environment.
IR describes the relationship between a dry sound and the reflected
sound from the boundaries in the scene. The reflected sound signal
depends on the scene geometry, scene materials, and source and listener
positions. IRs can be accurately simulated by solving the wave
equation [@FDTD]. However, solving a wave equation is not practical for
interactive applications because of its complexity. In our work, we
propose a learning-based IR generator (MESH2IR) that is capable of
approximating thousands of IRs per second for a given complex scene. Our
network can be formally described as: $$\begin{equation}
\label{overall_network}
\begin{aligned}[b]
   IR_{n} = \mathbf{N_{\theta_{1}}}(\mathbf{N_{\theta_{2}}}(M_{n}),SP_{n},LP_{n}),
\end{aligned}
\end{equation}$$ where $IR_{n}$ is the predicted IR for the given scene
$n$, $M_{n}$ is the 3D mesh representation of the scene simplified to
have around 2000 faces, $LP_{n}$ and $SP_{n}$ are listener location and
source location, respectively. $LP_{n}$ and $SP_{n}$ are represented
using a 3D vector. We simplify the 3D scene mesh with an arbitrary
number of faces to have a constant number of faces (2000 faces). We use
a mesh-based encoder network $\mathbf{N_{\theta_{2}}}$ to transform the
simplified 3D mesh into a latent vector of a latent space
(Figure [2](#graph_network){reference-type="ref"
reference="graph_network"}). $\mathbf{N_{\theta_{1}}}$ is a generator
network, and $\theta_{1}$ and $\theta_{2}$ are the trained network
parameters. Our overall network architecture is shown in
Figure [1](#full_architecture){reference-type="ref"
reference="full_architecture"}.

In recent years, cross-modal translation neural networks have gained
attention in computer vision. Several algorithms have been proposed for
translating video to audio [@video2audio], image to
audio [@image2reverb], text to image [@stackgan; @{stackgan++}], image
to mesh [@image2mesh], etc. In our work, we translate complex scenes
represented using meshes to acoustic IRs represented as audio signals.

## TRAINING DATASET

In our work, we train MESH2IR on the GWA IR dataset [@GWA] simulated
using a hybrid algorithm to generate realistic IRs on unseen complex
environments. The IRs in GWA are created by automatically calibrating
the ray energies simulated using the geometric acoustic
method [@pygsound] with the wave effects simulated using
finite-difference time-domain wave solver [@FDTD] to create high-quality
low-frequency and high-frequency wave effects. The GWA dataset consists
of 2 million IRs simulated on the indoor 3D environments represented as
meshes in the 3D-FRONT dataset. Out of the 2 million IRs, we train
MESH2IR on 200,000 IRs simulated in 5,000 different indoor environments
from 3D-FRONT dataset [@3dfront].

## MESH SIMPLIFICATION

The number of faces in the 3D-FRONT dataset varies from about 2000 faces
to 3 million faces. We initially simplify the meshes using a
quadratic-based edge collapse algorithm in PyMeshLab [@pymeshlab] to
have a fewer number of faces (i.e., 2000 faces). The edge collapse
algorithm makes sure that the approximation error between the original
mesh and the simplified mesh, in terms of Hausdorff distance is small.
Therefore the acoustic characteristics do not change due to the
simplification step.

Figure [3](#mesh_simplification){reference-type="ref"
reference="mesh_simplification"} depicts an example of the original
indoor 3D scene mesh and the simplified mesh using the quadratic edge
collapse algorithm. We can see that high-level details of the furniture
such as bed, pillow, settee, table and cupboard are preserved in the
simplified mesh, in addition to the scene geometry. In this example, the
simplified mesh has only 2% of faces of the original mesh.

<figure id="mesh_simplification" data-latex-placement="!h">

<figcaption>The indoor 3D scene is represented using an original mesh
with around 100,000 faces and a simplified mesh with 2,000 faces. We can
see that high-level details of the room geometry and furniture are
preserved in the simplified mesh. </figcaption>
</figure>

## MESH ENCODER

Our goal is to transform the simplified indoor scene meshes into a
low-dimensional latent space. The triangular meshes can be represented
as graph data. Therefore, we represent the 3D scene meshes as a graph
and use a graph network (Figure
 [2](#graph_network){reference-type="ref" reference="graph_network"}) to
reduce the dimension. Our graph network uses graph convolution (GCN)
layers [@GCN] to encode the topology information (i.e., edge
connectivity) and node features (i.e., vertex coordinates) of the graph.
We use graph pooling (gpool) layers [@Kpool1; @Kpool2] to reduce the
topology information of the graph.

The layer-wise propagation rule of multi-layer GCN [@GCN]:
$$\begin{equation}
\label{overall_networ3}
\begin{aligned}[b]
   X^{(l+1)} = \sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}X^{(l)}W^{(l)}),
\end{aligned}
\end{equation}$$ where $\hat{D}_{ii}  = \sum_{j} \hat{A}_{ij}$, and
$\hat{A}  = A + I$. $A$ is the adjacency matrix, $I$ is the identity
matrix and $W^{(l)}$ is a trainable weight matrix for layer $l$. The
feature matrices at layers $l$ and $l+1$ are $X^{(l)}$ and $X^{(l+1)}$,
respectively. The topology information is not modified in the GCN layer.

The gpool layer [@Kpool1] creates a new graph with high-level feature
encoding by choosing $K$ vertices from the original vertex set and
discarding the other vertices. Some edges are removed when discarding
vertices in the gpool layer. This results in some isolated vertices in
the graph. To address this problem, the gpool algorithm increases the
graph connectivity by calculating the square of the adjacency matrix
$A^{(l)}$ at layer $l$ and uses the new adjacency matrix $A_n^{(l)}$ for
gpool computation (Equation [\[gpool1\]](#gpool1){reference-type="ref"
reference="gpool1"}). $$\begin{equation}
\label{gpool1}
\begin{aligned}[b]
   A_n^{(l)} = A^{(l)} A^{(l)}.
\end{aligned}
\end{equation}$$

We calculate the channel-wise average (GAP) and the channel-wise maximum
(GMP) of the node features after each gpool layer and aggregate the GAP
and GMP values. We concatenate the aggregated GAP and aggregated GMP
values, pass them to linear layers, and get the mesh latent vector
($\pi_{M}$) (dimension=8) as the output.

## SCENE VECTOR EMBEDDING

We concatenate the mesh latent vector ($\pi_{M}$) with the source ($SP$)
and listener positions ($LP$) in 3D Cartesian space to generate scene
vector embedding $\pi_{A}$ of dimension 14
(Equation [\[embedd\]](#embedd){reference-type="ref"
reference="embedd"}). The mesh latent vector will be learned during
training. $$\begin{equation}
\label{embedd}
\begin{aligned}[b]
  \pi_{A}  = [\pi_{M} \; SP \; LP ].
\end{aligned}
\end{equation}$$

## IR REPRESENTATION & PREPROCESSING {#sec:IR_representation}

The IRs in the GWA dataset [@GWA] have a sampling rate of 48 kHz. We
downsample the IRs in the dataset to 16 kHz and crop the IRs.
Downsampling IRs allows us to maintain a longer duration of IRs in a
fixed-length input IR vector (3968 samples). The IRs with full duration
and the IRs cropped to have a duration of around 0.25 seconds give
similar performance in speech applications [@fast-rir].

**IR preprocessing:** The IRs in the GWA dataset have large variations
in standard deviation (STD) of the magnitude values ($10^{-12}$ to
$10^{-2}$). We noticed that it is hard for MESH2IR to learn from such
datasets with high dynamic ranges. To overcome this issue, we divide the
IRs by ten times the STD to create preprocessed IRs with the same STD of
0.1 (or any constant STD). We noticed that preprocessing IRs to have
constant STD improves the accuracy of MESH2IR (see
Section [4.1](#ablation:IR_pre){reference-type="ref"
reference="ablation:IR_pre"}).

To recover the IR with the original magnitude, we duplicate the STD of
the IRs 128 times and concatenate them at the end of the preprocessed
IRs. The concatenated IRs will have a length of 4096 (3968+128). The
convolution layers in MESH2IR calculate an average of 41 neighboring
sample values for each sample of the concatenated IRs and pass them to
the next layer. Therefore, concatenated STD values near the end of
preprocessed IR magnitudes in a concatenated IR will be corrupted when
we calculate the average and pass it to the next layers. We can recover
the STD values from the tail-end of the concatenated IR where all the 41
neighboring samples are STD.

## ENERGY DECAY RELIEF

The energy decay relief (EDR) obtained from the physics-based algorithm
contains enough information to be converted into an "equivalent impulse
response" [@energycurve1]. Therefore, we use EDR in the cost function of
our MESH2IR. Constraining the Generator network with additional
information (i.e., EDR) helps the model to converge easily.

The energy decay curve (EDC) describes the energy remaining in the IR
with respect to time [@EDC]. When compared to the IR itself, the EDC
decays more smoothly, and we can use it to measure IR acoustic metrics.
The generalized EDC for multiple frequency bands is called EDR [@EDR].
EDR is the total amount of energy remaining in the IR at time $t_n$
seconds in a frequency band centered at $f_k$ Hz:

$$\begin{equation}
\label{EDRelief}
\begin{aligned}[b]
  EDR(t_n,f_k)  = \sum_{m=n}^M |H(m,k)|^2.
\end{aligned}
\end{equation}$$

In Equation [\[EDRelief\]](#EDRelief){reference-type="ref"
reference="EDRelief"}, bin $k$ of the short-time Fourier transform at
time-frame $m$ is denoted by $H(m,k)$. $M$ is the total number of time
frames.

## MODIFIED CONDITIONAL GAN

Our MESH2IR passes the scene vector embedding $\pi_{A}$
(Equation [\[embedd\]](#embedd){reference-type="ref"
reference="embedd"}) to a one-dimensional modified conditional GAN
(CGAN) architecture proposed in FAST-RIR [@fast-rir] to generate a
single precise IR for the given indoor 3D scene. CGAN [@CGAN1; @CGAN2]
is conditioned on a random noise $z$ and on a condition $y$ to generate
multiple different outputs that satisfy the condition $y$. The modified
CGAN is only conditioned on $y$ to generate a single output.

MESH2IR consists of a generator network ($G_n$) and a discriminator
network ($D_n$), which are alternately trained at each iteration. We
train $G_n$ and $D_n$ using the IR samples from the data distribution
$p_{data}$. The objective function of our generator network consists of
modified CGAN error, EDR error, and mean square error (MSE). We train
the discriminator network with a modified CGAN objective function. For
each $\pi_{A}$, we use the corresponding IRs in the GWA dataset [@GWA]
as the ground truth when training our network.

#### **Modified CGAN Error (Generator Network):**

The CGAN error is used to generate IRs that are hard for the $D_n$ to
differentiate from the ground truth IRs. $$\begin{equation}
\label{CGAN_loss}
\begin{aligned}[b]
    \mathcal{L}_{CGAN} = \mathbb{E}_{\pi_{A} \sim p_{data}}[\log(1 - D_{N}(G_{N}(\pi_{A})))].
\end{aligned}
\end{equation}$$

#### **EDR Error:**

For each $\pi_{A}$, we calculate the EDR of the generated IR using our
MESH2IR ($E_N$) and the ground truth IR ($E_G$). We calculate EDR at a
set of frequency bands with center frequencies at 125 Hz, 250 Hz, 500
Hz, 1000 Hz, 2000 Hz, and 4000 Hz ($B$). We compare the $E_N$ and $E_G$
for each sample ($s$) as follows:

$$\begin{equation}
\label{edr_loss}
\begin{aligned}[b]
    \mathcal{L}_{EDR} = \mathbb{E}_{\pi_{A} \sim p_{data}} [\mathbb{E}_{b \sim B}[\mathbb{E}[(E_{N}(\pi_{A},b,s) - E_{G}(\pi_{A},b,s))^{2}]]].
\end{aligned}
\end{equation}$$

The signal energy in high-frequency components of EDR is high when
compared with the signal energy in low-frequency components of EDR in
the training IR dataset [@GWA]. Therefore, we give more weight to
low-frequency components of EDR.

#### **MSE Error:** 

For each $\pi_{A}$, we compare the IR generated from MESH2IR ($I_N$)
with the ground truth IR ($I_G$). We calculate the squared difference of
each sample ($s$) in $I_N$ and $I_G$ as follows:

$$\begin{equation}
\label{mse_loss}
\begin{aligned}[b]
    \mathcal{L}_{MSE} = \mathbb{E}_{\pi_{A} \sim p_{data}}[\mathbb{E}[(I_{N}(\pi_{A},s) - I_{G}(\pi_{A},s))^{2}]].
\end{aligned}
\end{equation}$$

The generator network ($G_N$) and the discriminator network ($D_N$) are
trained to compete each other by minimizing the generator objective
function $\mathcal{L}_{G_{N}}$
(Equation [\[generator_loss\]](#generator_loss){reference-type="ref"
reference="generator_loss"}) and maximizing the discriminator objective
function $\mathcal{L}_{D_{N}}$
(Equation [\[discriminator_loss\]](#discriminator_loss){reference-type="ref"
reference="discriminator_loss"} ). In
Equation [\[generator_loss\]](#generator_loss){reference-type="ref"
reference="generator_loss"}, the relative importance of the EDR error
and MSE error are controlled using $\lambda_{EDR}$ and $\lambda_{MSE}$
respectively.

$$\begin{equation}
\label{generator_loss}
\begin{aligned}[b]
    \mathcal{L}_{G_{N}} = \mathcal{L}_{CGAN} + \lambda_{EDR} \; \mathcal{L}_{EDR} + \lambda_{MSE} \; \mathcal{L}_{MSE}.
\end{aligned}
\end{equation}$$

$$\begin{equation}
\label{discriminator_loss}
\begin{aligned}[b]
    \mathcal{L}_{D_{N}} = \mathbb{E}_{(I_{G},\pi_{A}) \sim p_{data}}[\log(D_{N}(I_{G}(\pi_{A})))] \\
    + \mathbb{E}_{\pi_{A} \sim p_{data}}[\log(1 - D_{N}(G_{N}(\pi_{A})))].
\end{aligned}
\end{equation}$$

## IMPLEMENTATION DETAILS

#### **Network Architecture:** 

We adapt the graph encoder network in the PyG official
repository [@web-pyg] and modify the network to encode indoor 3D scene
meshes to the latent space
(Figure [2](#graph_network){reference-type="ref"
reference="graph_network"}). Our gpool layer keeps 60% of the original
number of vertices in each layer. We use the Generator ($G_N$) network
and the Discriminator network ($D_N$) architectures proposed in
FAST-RIR [@fast-rir] and modify their cost functions. We extend the
$G_N$ architecture proposed in FAST-RIR by adding our mesh encoder
network (Figure [1](#full_architecture){reference-type="ref"
reference="full_architecture"}).

#### **Training:**

We trained $G_N$ and $D_N$ using the RMSprop optimizer with a batch size
of 256. The $G_N$ is iterated 3 times for every $D_N$ iteration.
Initially, we start with the learning rate of 8x$10^{-5}$ and decay the
learning rate by 85% of the original value for every 7 epochs. We
trained our network for 150 epochs. We published our code for
reproducibility at github [@web-MESH2IR].

:::: table*
::: tabular
L0.1L0.15L0.15L0.15L0.15L0.15 Benchmark Scene &

![image](benchmarks/1aa91215-cba7-4c40-8b37-6b21584b5924.png){width="1.1600 in"}

&

![image](benchmarks/1b22c3c0-3805-406c-91f9-d35d922d5914.png){width="1.1600 in"}

&

![image](benchmarks/27982f11-7de0-4f79-b89f-9f2354856f24.png){width="1.1600 in"}

&

![image](benchmarks/75c62dac-2d9c-4d49-b43a-b66986acddd4.png){width="1.1600 in"}

&

![image](benchmarks/9ac25401-5aa3-426d-aa37-a04d0b7d5448.png){width="1.1600 in"}

\
\# faces & 185,319 & 92,422 & 181,536 & 253,684 & 282,375\
Ground truth IR & \[1.00001\]

![image](benchmarks/2.7-gd.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/5.9-gd.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/8.2-gd.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/0.6-gd.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/7.5-gd.png){width="1.1600 in"}

\
Our Mesh2IR & \[1.00001\]

![image](benchmarks/2.7-pred.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/5.9-pred.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/8.2-pred.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/0.6-pred.png){width="1.1600 in"}

& \[1.00001\]

![image](benchmarks/7.5-pred.png){width="1.1600 in"}

\
$T_{60}$ error &2.7% &5.9% &8.2% &0.6% &7.5%\
DRR error & 2.6% &9.4% & 1.4% &1.9% &9.3%\
EDT error & 2.9% & 2.1% & 2.1% & 1.7% & 0.9%\
:::
::::

# ABLATION EXPERIMENTS {#ablation}

We perform an ablation study to analyze the importance of our IR
preprocessing approach
(Section [3.6](#sec:IR_representation){reference-type="ref"
reference="sec:IR_representation"}) and to find an efficient way of
adding EDR to the cost function. We quantitatively evaluate different
variations of our network by calculating the mean absolute difference of
different acoustic metrics of the generated IRs and the ground truth
IRs. The acoustic characteristics of the 3D indoor environment are
described using acoustic metrics [@augment_IR]. We also measure the mean
square error (MSE) of the generated IRs and the ground truth IRs
(Equation [\[mse_loss\]](#mse_loss){reference-type="ref"
reference="mse_loss"}). We train all the variations of our network with
200K IRs from 5000 different indoor 3D scene meshes and generate 11K
testing IRs from 600 scene meshes not used during training. We use
commonly-used acoustic metrics such as reverberation time ($T_{60}$),
early-decay-time (EDT), and direct-to-reverberant ratio (DRR) in our
experiment. The time taken to decay the sound pressure by 60 decibels
(dB) is called $T_{60}$. DRR is the ratio of the sound pressure level of
the direct sound to the sound pressure level of reflected
sound [@drr_book]. EDT is six times the time taken for the sound source
to decay by 10 dB, and it depends on the type and location of the sound
source.

## IR PREPROCESSING {#ablation:IR_pre}

We evaluate the contribution of our IR preprocessing approach discussed
in Section [3.6](#sec:IR_representation){reference-type="ref"
reference="sec:IR_representation"}. We train our MESH2IR on IRs without
IR preprocessing (MESH2IR-UNPROCESSED) and generate IRs from our trained
network. Figure [4](#ir_pre_process){reference-type="ref"
reference="ir_pre_process"} shows a ground truth IR and the IR generated
from MESH2IR-UNPROCESSED. We can see that MESH2IR-UNPROCESSED generates
noisy output. In
Table [\[tab:MESH2IR_time_domain\]](#tab:MESH2IR_time_domain){reference-type="ref"
reference="tab:MESH2IR_time_domain"} we can see that MESH2IR estimates
the IRs for the given 3D scene to a greater extent.

<figure id="ir_pre_process" data-latex-placement="!h">

<figcaption>The ground truth IR, and the IR predicted by MESH2IR trained
with an unprocessed IR training set (MESH2IR-UNPROCESSED). We can see
that MESH2IR-UNPROCESSED gives noisy output.</figcaption>
</figure>

## ENERGY DECAY RELIEF (EDR) {#ablation:EDR}

To evaluate the importance and the efficient way of adding EDR to the
cost functions, we train and compare two different variations of our
MESH2IR network.\

**Variation 1 (MESH2IR-NO-EDR) :** We evaluate the role played by the
EDR loss in the generator network by training a variant of MESH2IR
without the EDR loss
(Equation [\[edr_loss\]](#edr_loss){reference-type="ref"
reference="edr_loss"}).\

**Variation 2 (MESH2IR-D-EDR) :** Instead of training the Discriminator
network to discriminate between generated IRs and ground truth IRs of
dimension 3968 x 1, we train the Discriminator network to discriminate
between the EDR of the generated IRs and the ground truth IRs. EDR is
calculated at six octave bands with center frequencies at 125 Hz, 250
Hz, 500 Hz, 1000 Hz, 2000 Hz and 4000 Hz. The EDR of an IR has a
dimension of 3986x6. We also train the Generator network with the EDR
loss.\

From Table [1](#tab:acoustic_param){reference-type="ref"
reference="tab:acoustic_param"}, we can see that adding EDR in the
generator loss function improves the $T_{60}$, EDT, and MSE. Training
the Discriminator network with IRs (MESH2IR) gives a similar performance
to passing EDR of the IRs (MESH2IR-D-EDR).

::: {#tab:acoustic_param}
+----------------+-----------------------------------------------+---------------------+
| **IR Dataset** | **Mean Absolute Error $\downarrow$**          | **MSE (x            |
|                |                                               | $\mathbf{10^{-4}})$ |
|                |                                               | $\downarrow$**      |
+:===============+:=================:+:===========:+:===========:+:===================:+
| 2-4            | $\mathbf{T_{60}}$ | **DRR**     | **EDT**     |                     |
+----------------+-------------------+-------------+-------------+---------------------+
| MESH2IR-NO-EDR | 0.16              | **2.68**    | 0.23        | 1.44                |
+----------------+-------------------+-------------+-------------+---------------------+
| MESH2IR-D-EDR  | **0.13**          | 2.74        | **0.22**    | **1.23**            |
+----------------+-------------------+-------------+-------------+---------------------+
| **MESH2IR**    | **0.13**          | 2.72        | **0.22**    | **1.23**            |
+----------------+-------------------+-------------+-------------+---------------------+

: Mean absolute error of the acoustic metrics and mean square error of
the generated IRs from MESH2IR-NO-EDR, MESH2IR-D-EDR and MESH2IR. The
acoustic metrics used in our experiments are reverberation time
($\mathbf{T_{60}}$) measured in seconds, direct-to-reverberant ratio
(DRR) measured in decibels and early-decay-time (EDT) measured in
seconds. We can see that overall MESH2IR gives the least error. The best
results of each metric are \"bolded\".
:::

## POWER SPECTRUM {#sec:power_spectrum}

The power spectrum describes the energy distribution of the IR in the
frequency components that compose the waveform. In
Figure [5](#power_spectrum){reference-type="ref"
reference="power_spectrum"}, we compare the power spectrum of ground
truth IRs with the power spectrum of the predicted IR using MESH2IR,
MESH2IR-D-EDR, and MESH2IR-NO-EDR. In this example, we placed the
listener 1m and 8m away from the source. We can see that in both cases
the power spectrum of MESH2IR is closer to GWA when compared with other
variants of our approach.

<figure id="power_spectrum" data-latex-placement="t">

<figcaption>The power spectrum of IRs generated using our proposed
MESH2IR, MESH2IR-D-EDR, MESH2IR-NO-EDR, and the ground truth IRs from
the GWA dataset. We can see that the power spectrum of MESH2IR is
closest to the power spectrum of GWA. </figcaption>
</figure>

# ACOUSTIC EVALUATION

In this section, we evaluate the accuracy, power spectrum and runtime of
our MESH2IR network. We evaluate the accuracy of our MESH2IR network by
comparing the relative acoustic metric error values of the IRs predicted
from MESH2IR on indoor 3D scenes not used during training and the ground
truth IRs from the GWA dataset [@GWA]. We compare the run-time of
MESH2IR network with a state-of-the-art geometric acoustic
simulator [@pygsound].

## ACCURACY ANALYSIS

To evaluate the robustness of our MESH2IR, we selected ground truth IRs
from the GWA dataset [@GWA] from 5 different indoor 3D scenes with
different levels of complexity (the number of faces in a 3D scene mesh).
The selected IRs have a large variation in magnitudes and shapes. The
distance between the listener and the source position varies from 3.6m
to 10.5m. We predicted IRs corresponding to the ground truth IRs in 3D
scenes using our MESH2IR. We evaluate the accuracy of our prediction
using relative $T_{60}$ error, relative DRR error, and relative EDT
error.
Table [\[tab:MESH2IR_time_domain\]](#tab:MESH2IR_time_domain){reference-type="ref"
reference="tab:MESH2IR_time_domain"} shows the ground truth IRs,
predicted IRs and the accuracy of our prediction. We can see that our
approach can predict the IRs with less than 10% $T_{60}$ and DRR errors,
and less than 3% EDT error, which are significantly lower than their
just-noticeable-differences
(JNDs) [@blevins2013quantifying; @del2022study; @werner2014adjustment].

## RUNTIME {#sec:runtime}

We compare the runtime for generating 30,000 IRs from our MESH2IR with
the time required to generate these samples from a geometric-based
acoustic simulator (GA) [@pygsound] and FAST-RIR [@fast-rir]. The
runtime of GA depends on the complexity of the scene, and FAST-RIR is
only capable of generating IRs for empty shoe-box-shaped medium-sized
rooms (i.e., room dimensions varying from \[8m,6m,2.5m\] to
\[11m,8m,3.5m\]). We compare the runtime of GA and FAST-RIR given in
Ratnarajah et al.  [@fast-rir] with the runtime of MESH2IR for
generating IRs for furnished indoor 3D scene meshes in the 3D-Front
dataset [@3dfront]. We use an Intel(R) Xenon(R) CPU E52699 v4 @ 2.20 GHz
and a GeForce RTX 2080 Ti GPU for our evaluation.

For a fair comparison with GA and FAST-RIR methods, which take 3D room
dimensions as inputs, we did not consider the time taken for mesh
simplification and mesh-to-graph conversion in
Table [2](#tab:runtime){reference-type="ref" reference="tab:runtime"}.
On average, mesh simplification takes around 7.5 seconds and
mesh-to-graph conversion takes 0.04 seconds. From
Table [2](#tab:runtime){reference-type="ref" reference="tab:runtime"},
we can see that MESH2IR is more than 200 times faster than GA on a CPU.
MESH2IR is optimized to run on GPUs and supports batch parallelization.
MESH2IR is slower than FAST-RIR because we encode a 3D scene mesh
represented using a graph to a mesh latent vector using our graph neural
network (Figure  [2](#graph_network){reference-type="ref"
reference="graph_network"}). To generate thousands of IRs for a given
furnished indoor 3D scene, we perform mesh simplification, mesh-to-graph
conversion, and mesh encoding only once.

We have shown the average time taken for mesh encoding (MESH2IR\[Mesh
Encoder\]) and IR generation using the encoded mesh (MESH2IR\[IR
Generator\]) separately in Table [2](#tab:runtime){reference-type="ref"
reference="tab:runtime"}. Our MESH2IR can generate more than 10,000 IRs
per second on a single GPU for a given furnished indoor 3D scene, and
the runtime of MESH2IR is stable irrespective of the scene.

# APPLICATIONS

We demonstrate the benefit of our MESH2IR for several downstream speech
applications - neural sound rendering, speech dereverberation and
reverberant speech separation. For a fair comparison between different
methods, we generate 11K IRs from 600 scene meshes not used during
training our MESH2IR using GA [@pygsound], GWA [@GWA], MESH2IR-D-EDR,
and MESH2IR. We generate reverberant speech using the 11K IRs, and train
speech dereverberation and speech separation methods. Reverberant speech
can be generated from a dry sound source and an IR via a convolution:
$$\begin{equation}
x(t) = s(t)*r(t) %+ n(t)
\vspace{-0.1cm}
\end{equation}$$ where $x(t)$ is the reverberant speech signal, $r(t)$
is the IR, $s(t)$ is the dry speech signal.

::: {#tab:runtime}
  **IR Generator**               **Hardware**      **Avg time**       **Scene Type**     
  ----------------------------- -------------- --------------------- ---------------- -- --
  GA [@pygsound]                     CPU              30.05s              Simple         
  **MESH2IR**                      **CPU**           **0.13s**         **Complex**       
  MESH2IR(Batch Size 1)              GPU          1.32x$10^{-2}$s        Complex         
  FAST-RIR(Batch Size 128)           GPU          5.9x$10^{-5}$s          Simple         
  **MESH2IR(Batch Size 128)**      **GPU**      **2.6x$10^{-3}$s**     **Complex**       
  **MESH2IR\[Mesh Encoder\]**      **GPU**      **2.57x$10^{-3}$s**    **Complex**       
  **MESH2IR\[IR Generator\]**      **GPU**      **7.4x$10^{-5}$s**     **Complex**       

  : The runtime of a geometric acoustic simulator (GA) [@pygsound],
  FAST-RIR [@fast-rir], and our MESH2IR. MESH2IR is an extension of
  FAST-RIR to generate IRs for complex indoor scenes. We can see that
  the runtime of MESH2IR is higher than FAST-RIR because we use a
  graph-based network to process the mesh. MESH2IR still outperforms GA
  on a single CPU.
:::

## NEURAL SOUND RENDERING

Our MESH2IR is the first neural-network-based approach to predict IRs
from a given 3D scene mesh at interactive rates. Given this advantage,
MESH2IR can be applied to general 3D scenes to enable real-time neural
sound rendering, without pre-computation on new scenes. For single IR
updates in dynamic scenes, MESH2IR can operate at more than 100 frames
per second (fps), which is significantly beyond the requirement for
interactive applications (e.g., 10 fps). We demonstrate its sound
rendering quality in our supplemental video [^1].

## SPEECH DEREVERBERATION {#enhancement}

Speech dereverberation is the process of obtaining reverb-free speech
from reverberant speech. We train speech dereverberation models using
data generated from different synthetic IR generation methods and
compare their performance on data generated from recorded IRs. We test
the models on IRs from the MIT IR dataset [@traer2016statistics], the
BUTReverb dataset [@8717722], and the RWCP Aachen IR dataset
[@nakamura1999sound; @jeub2009binaural]. For all experiments, we train
the SkipConvNet [@kothapally2020skipconvnet] model with its default
parameters. We use the metric speech-to-reverberation modulation energy
ratio (SRMR) [@falk2010non] to measure the improvement obtained by the
dereverberation process. Higher SRMR implies higher speech quality and
lower reverberation effects. We also report the SRMR of the baseline
where we do not apply any dereverberation (Reverb).

In Table [3](#tab:realResultsSE){reference-type="ref"
reference="tab:realResultsSE"}, we test the models on reverberant data
generated from recorded IRs. Across all IR datasets, we see that the
SRMR of our MESH2IR model is similar to the SRMR of the GWA IRs [@GWA].
We also see that MESH2IR outperforms the GA [@pygsound] method, which
only generates IRs using geometric simulations (less accurate than IRs
generated from the GWA dataset). The SRMR of MESH2IR is closer to GWA
IRs when compared with MESH2IR-D-EDR. Therefore, MESH2IR generates more
accurate IRs when compared with MESH2IR-D-EDR.

::: {#tab:realResultsSE}
+----------------------+--------------------------------------------+
| **Training Dataset** | **SRMR**                                   |
+:=====================+:========:+:=============:+:===============:+
| 2-4                  | **MIT**  | **BUTReverb** | **RWCP Aachen** |
+----------------------+----------+---------------+-----------------+
| Reverb               | 7.35     | 3.14          | 5.16            |
+----------------------+----------+---------------+-----------------+
| GA [@pygsound]       | 6.39     | 3.74          | 4.83            |
+----------------------+----------+---------------+-----------------+
| GWA [@GWA]           | **7.67** | **4.6**       | **6.14**        |
+----------------------+----------+---------------+-----------------+
| **MESH2IR-D-EDR**    | 6.18     | 3.29          | 4.32            |
+----------------------+----------+---------------+-----------------+
| **MESH2IR (ours)**   | **7.82** | **4.27**      | **5.88**        |
+----------------------+----------+---------------+-----------------+

: Speech dereverberation results are obtained when training data is
generated by different synthetic IR generation methods. Testing is done
on reverberant data synthesized from IRs present in three different
datasets containing recorded IRs collected in a variety of environments.
Higher SRMR is better.
:::

## SPEECH SEPARATION {#seperarion}

Speech separation, also referred to as the cocktail party problem, is
the process of separating a mixture of speech signals into its
constituent speech signals. We check the performance of different
synthetic IR generation methods on the task of reverberant speech
separation - separating a reverberant mixture into its constituent
reverberant speech signals. The dry speech signals and mixtures from the
Libri2Mix [@cosentino2020librimix] dataset are convolved with IRs to
generate the reverberant data. We train the DPRNN-TasNet [@luo2020dual]
model for all speech separation experiments. We utilize the default
implementation provided by the Asteroid [@pariente2020asteroid]
framework. The model is tested on reverberant mixtures created from real
recordings obtained from four different room configurations in the
VOiCES [@richey2018voices] dataset. We clearly see that our MESH2IR
performs similar to GWA [@GWA], and outperforms GA method [@pygsound]
and MESH2IR-D-EDR.

+-------------------+---------------------------------------------------+
| **Training        | **SI-SDRi**                                       |
| Dataset**         |                                                   |
+:==================+:==========:+:==========:+:==========:+:==========:+
| 2-5               | **Room** 1 | **Room 2** | **Room 3** | **Room 4** |
+-------------------+------------+------------+------------+------------+
| GA [@pygsound]    | 2.26       | 2.22       | 1.33       | 2.35       |
+-------------------+------------+------------+------------+------------+
| GWA [@GWA]        | **4.75**   | **4.75**   | **2.41**   | **4.91**   |
+-------------------+------------+------------+------------+------------+
| **MESH2IR-D-EDR** | 4.68       | 4.35       | 1.87       | 4.72       |
+-------------------+------------+------------+------------+------------+
| **MESH2IR         | **4.91**   | **4.89**   | **2.54**   | **5.13**   |
| (ours)**          |            |            |            |            |
+-------------------+------------+------------+------------+------------+

: Speech separation results in the presence of reverberation are shown
below. We report the improvement in the Scale-Invariant Signal
Distortion Ratio (SI-SDRi) over the reverberant mixture. Higher Si-SDRi
is better. We report performance on reverberant mixtures generated in
four different room configurations present in the VOiCES dataset.

# CONCLUSION AND FUTURE WORK {#sec:conclusion}

We present a novel neural-network-based MESH2IR architecture to generate
thousands of IRs for a given furnished indoor 3D scene on the fly. The
IR generation speed of our MESH2IR is constant within a given complex 3D
scene, irrespective of the complexity of the scene. Our MESH2IR can
generate thousands of IRs per second for a given 3D scene. We show that
the IRs predicted by our MESH2IR in unseen indoor 3D scenes are highly
similar to the ground truth IRs generated from the GWA dataset, which is
used to train our MESH2IR.

Our work has some limitations. One is we cannot control the
characteristics of the scene materials such as the amount of sound
absorption and scattering, which can affect the overall amplitude of the
IR. Efficiently inputting the characteristics of scene material to our
MESH2IR may improve the accuracy of IR generation. In addition, while
MESH2IR can handle moving sound by simulating many IRs according to a
trajectory, when objects in the scene moves, the scene representation
changes, which incurs additional encoding time for MESH2IR, making it
less efficient in highly dynamic scenes. In the future, we would like to
integrate our MESH2IR into game engines and other interactive
applications, and evaluate its benefits.

[^1]: <https://anton-jeran.github.io/M2IR/>
