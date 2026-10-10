# Privacy-Preserving Deep Learning Toolkit

A collection of research implementations for privacy-preserving deep
learning in vehicular edge computing and online AI services. The
projects use techniques including **additive secret sharing** and
**secure multi-party computation (MPC)** to support inference and data
processing without requiring a single edge server to access the original
sensitive input.

## Projects

This repository contains four project areas:

1.  [Privacy-Preserving Object
    Detection](#1-privacy-preserving-object-detection) --- P3 / vePOD
2.  [Privacy-Preserving Data
    Sharing](#2-privacy-preserving-data-sharing) --- PRECISE
3.  [Privacy-Preserving Driver
    Monitoring](#3-privacy-preserving-driver-monitoring-system) --- PDMS
4.  [Privacy-Preserving GPT](#4-privacy-preserving-gpt) --- PPGPT

> **Note:** The project descriptions and results below summarize the
> accompanying project documentation and publications. Check each
> project directory for its implementation details, dependencies, and
> instructions for running the code.

------------------------------------------------------------------------

## 1. Privacy-Preserving Object Detection

**Frameworks:** P3 and vePOD

Autonomous vehicles use deep neural networks (DNNs) for perception tasks
such as object detection. Offloading these compute-intensive tasks to
roadside edge servers can reduce onboard computation, but sending raw
sensor data to potentially untrusted servers creates privacy risks.

P3/vePOD uses **additive secret sharing**: input data or intermediate
features are divided into random shares and processed across multiple
edge servers. Partial outputs are combined to reconstruct the detection
result, without any one participating server needing to see the complete
input.

### Key features

-   **Secure CNN operations:** Secure implementations of convolution,
    fully connected layers, ReLU, Leaky ReLU, max/average pooling, ROI
    pooling, and region proposal network (RPN) operations.
-   **PP-Faster R-CNN:** Uses feature-level secret sharing. The vehicle
    computes the backbone and RPN locally, then shares feature maps and
    region proposals for secure processing on edge servers.
-   **PP-YOLO:** Uses image-level secret sharing, with the network
    operating on shares of the input image.
-   **Privacy-aware edge inference:** Designed to protect sensor inputs
    and detection outputs from individual edge servers.

### Results reported in the project documentation

  Model            Original mAP   Privacy-preserving mAP
  -------------- -------------- ------------------------
  Faster R-CNN            78.5%                    78.5%
  YOLOv3                  72.1%                    72.1%

The documentation reports the following inference times for a two-edge
setup:

-   **PP-YOLO:** approximately 3.08 s (original) and 80.7 s
    (privacy-preserving).
-   **PP-Faster R-CNN:** approximately 14 s (original) and 44.1 s
    (privacy-preserving).
-   The reported speedup over the homomorphic-encryption baseline
    CryptoNets is 8.8×--83.3×.

### Publication

-   Tianyu Bai, Danyang Shao, Ying He, Song Fu, and Qing Yang. "P3: A
    Privacy-Preserving Perception Framework for Building Vehicle-Edge
    Perception Networks Protecting Data Privacy." *32nd IEEE
    International Conference on Computer Communications and Networks
    (ICCCN)*, July 2023.
-   Tianyu Bai, Song Fu, and Qing Yang. "Privacy-Preserving Object
    Detection with Secure Convolutional Neural Networks for Vehicular
    Edge Computing." *Future Internet*, 14(11):316, October 2022.

------------------------------------------------------------------------

## 2. Privacy-Preserving Data Sharing

**Framework:** PRECISE

PRECISE (Privacy Preserving Data Sharing with Segmentation and
Inpainting for CAV) supports **user-defined privacy protection** when
connected autonomous vehicles share environmental data. Users can
specify object classes they want to keep private, such as faces, license
plates, or house numbers.

The framework combines secure segmentation, image inpainting, edge
computing, and additive secret sharing. The goal is to remove selected
sensitive objects and reconstruct the surrounding image while avoiding
disclosure of the original image to an individual edge server.

### Key features

-   **Secure SegNet:** Performs pixel-level segmentation to identify
    sensitive objects. The documented architecture is based on a VGG16
    encoder-decoder with 13 convolutional and 13 transpose-convolutional
    layers.
-   **Secure Context Encoder:** Inpaints regions where sensitive objects
    have been removed. The documented model uses an encoder with five
    convolutional layers and a decoder with five transpose-convolutional
    layers.
-   **Secure CNN operations:** Includes secure Leaky ReLU, max pooling,
    batch normalization, and max unpooling.
-   **Secret sharing and edge processing:** The sender splits image data
    into shares for processing by separate edge servers.
-   **User-defined privacy:** Supports configurable protection of
    selected object classes before data sharing.

### Execution times reported in the project documentation

  Component                       Execution time
  ----------------------------- ----------------
  Secure SegNet                           3.47 s
  Secure Context Encoder                  0.99 s
  Share generation/encryption             0.66 s
  Result combination                     0.002 s

The documentation reports comparisons with other secure segmentation
approaches, including BUNET (1078 s) and HSI (46 s). These figures are
reported results from the described test setup.

### Publication

Tianyu Bai, Qing Yang, and Song Fu. "User-Defined Privacy Preserving
Data Sharing for Connected Autonomous Vehicles Utilizing Edge
Computing." *The Eighth ACM/IEEE Symposium on Edge Computing (SEC '23)*,
December 2023.

DOI: [10.1145/3583740.3628436](https://doi.org/10.1145/3583740.3628436)

``` bibtex
@inproceedings{bai2023precise,
  author    = {Bai, Tianyu and Yang, Qing and Fu, Song},
  title     = {User-Defined Privacy Preserving Data Sharing for Connected Autonomous Vehicles Utilizing Edge Computing},
  booktitle = {The Eighth ACM/IEEE Symposium on Edge Computing (SEC '23)},
  year      = {2023},
  month     = {December},
  address   = {Wilmington, DE, USA},
  publisher = {ACM},
  doi       = {10.1145/3583740.3628436}
}
```

------------------------------------------------------------------------

## 3. Privacy-Preserving Driver Monitoring System

**Framework:** PDMS

Driver monitoring systems can help detect driver inattention, fatigue,
and other states, but processing video locally can strain vehicle
resources. Sending video to cloud services can expose sensitive facial
and behavioral information.

PDMS uses additive secret sharing and privacy-preserving Transformer
models to process driver video frames across edge servers. The design
aims to protect raw video during transmission and inference while
supporting tasks such as gaze estimation.

### Key features

-   **Privacy-preserving Vision Transformer (ViT):** Performs gaze
    estimation on secret shares.
-   **Privacy-preserving preprocessing:** Integrates privacy-preserving
    YOLO and FlowNet to identify eye regions and motion and to filter
    redundant frames.
-   **Secure Transformer operations:** Includes secure multiplication
    using Beaver's triples, secure GELU and ReLU activations, and
    optimized layer normalization.
-   **Edge computing:** Moves computationally intensive inference to
    edge servers while the vehicle retains control of the original data
    and reconstruction of the result.

### Evaluation described in the documentation

The reported testbed included a Polaris GEM e4 electric vehicle with an
NVIDIA Jetson AGX Xavier and AMD Ryzen 7 edge servers. The Gaze360 and
LISA Gaze datasets were used.

The documentation reports these execution-time figures:

  Component                                            Time per frame
  ------------------------------------------- -----------------------
  Privacy-preserving convolutional layers                       70 ms
  Privacy-preserving deconvolutional layers                     36 ms
  Privacy-preserving GELU activation                          1830 ms
  Total before optimization                     approximately 3472 ms
  Optimized total (128 × 128 input + GPU)       approximately 1690 ms

The documentation identifies secure GELU as a major performance
bottleneck and reports approximately 51% latency reduction after the
described optimizations.

### Publication

Tianyu Bai, Danyang Shao, Ying He, Qing Yang, Yunhe Feng, and Song Fu.
"Privacy-Preserving Driver Monitoring on the Edges: Transformer-Based
Processing of Secret Shares from Video Streams." *ACM Transactions on
Internet of Things*, Vol. 7, Issue 3, pp. 1--31, April 2026.

DOI: [10.1145/3777384](https://doi.org/10.1145/3777384)

``` bibtex
@article{bai2026privacy,
  title     = {Privacy-Preserving Driver Monitoring on the Edges: Transformer-Based Processing of Secret Shares from Video Streams},
  author    = {Bai, Tianyu and Shao, Danyang and He, Ying and Yang, Qing and Feng, Yunhe and Fu, Song},
  journal   = {ACM Transactions on Internet of Things},
  volume    = {7},
  number    = {3},
  pages     = {1--31},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3777384}
}
```

------------------------------------------------------------------------

## 4. Privacy-Preserving GPT

**Framework:** PPGPT

Online large language model (LLM) services may require users to send
potentially sensitive prompts to remote servers. PPGPT explores
privacy-preserving Transformer inference using additive secret sharing
and secure multi-party computation.

User inputs are split into random shares on the client side. Separate
servers process the shares using secure Transformer operations, and the
user reconstructs the final output from the returned partial results.

### Key features

-   **Additive secret sharing:** Splits user input into shares so that
    no single server holds the complete raw input.
-   **Secure Transformer layers:** Implements privacy-preserving
    versions of key Transformer operations.
-   **Secure GELU:** Uses Taylor-series approximation and Beaver's
    triples for secure nonlinear activation.
-   **Secure layer normalization:** Uses an optimized protocol intended
    to reduce secure multiplication overhead.
-   **Secure Softmax:** Computes probabilities using polynomial
    transformations.
-   **Client-side reconstruction:** Returns partial output shares to the
    user for reconstruction.

### Execution times reported in the project documentation

  Layer                        MiniGPT     PPGPT
  -------------------------- --------- ---------
  Linear                        3.7 ms   8.28 ms
  Layer normalization           8.7 ms   0.756 s
  GELU                          6.2 ms    0.32 s
  Dropout                       1.2 ms    1.9 ms
  Softmax                       0.4 ms     0.3 s
  Masked attention                1 ms   1.78 ms
  Attention multiplication      2.6 ms    0.37 s
  Total inference              23.8 ms   1.978 s

The documentation reports that PPGPT preserves output fidelity while
introducing additional computation time for secure operations.

### Publication

Tianyu Bai, Yunhe Feng, and Song Fu. "Safeguarding User Data Privacy in
Online Large Language Model Services." *Journal of Systems
Architecture*, Vol. 168, pp. 1--33, September 2025.

DOI:
[10.1016/j.sysarc.2025.103555](https://doi.org/10.1016/j.sysarc.2025.103555)

``` bibtex
@article{bai2025safeguarding,
  title     = {Safeguarding User Data Privacy in Online Large Language Model Services},
  author    = {Bai, Tianyu and Feng, Yunhe and Fu, Song},
  journal   = {Journal of Systems Architecture},
  volume    = {168},
  pages     = {1--33},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.sysarc.2025.103555}
}
```

------------------------------------------------------------------------

## Technical Approach

Across these projects, the main privacy-preserving approach is to
distribute computations over **additive secret shares** rather than
exposing complete raw inputs to a single edge server. Depending on the
application, the projects apply this approach to image pixels,
intermediate CNN features, video frames, or LLM inputs.

Secure nonlinear operations---such as ReLU, GELU, pooling, Softmax, and
layer normalization---are important components because they are more
difficult to compute efficiently over secret shares than linear
operations.

## Repository Structure

The repository is organized into four project directories. The links
below use the directory names in the current repository:

-   [Object detection](./faster%20rcnn%20secure%20object%20detection/)
-   [Data sharing](./secure%20data%20sharing/)
-   [Driver monitoring](./secure%20driver%20monitoring%20system/)
-   [Privacy-preserving GPT](./secure%20gpt/)

If you rename these folders, update the links above to match the new
names.

## Publications at a Glance

1.  **P3:** Bai et al., "P3: A Privacy-Preserving Perception Framework
    for Building Vehicle-Edge Perception Networks Protecting Data
    Privacy." ICCCN, 2023.
2.  **vePOD:** Bai, Fu, and Yang, "Privacy-Preserving Object Detection
    with Secure Convolutional Neural Networks for Vehicular Edge
    Computing." *Future Internet*, 2022.
3.  **PRECISE:** Bai, Yang, and Fu, "User-Defined Privacy Preserving
    Data Sharing for Connected Autonomous Vehicles Utilizing Edge
    Computing." SEC, 2023.
    [DOI](https://doi.org/10.1145/3583740.3628436)
4.  **PDMS:** Bai et al., "Privacy-Preserving Driver Monitoring on the
    Edges: Transformer-Based Processing of Secret Shares from Video
    Streams." *ACM Transactions on Internet of Things*, 2026.
    [DOI](https://doi.org/10.1145/3777384)
5.  **PPGPT:** Bai, Feng, and Fu, "Safeguarding User Data Privacy in
    Online Large Language Model Services." *Journal of Systems
    Architecture*, 2025.
    [DOI](https://doi.org/10.1016/j.sysarc.2025.103555)

## Citation

If you use a specific component, please cite the corresponding
publication listed in that project's section.

## License

Please add the appropriate license for this repository if you intend to
distribute the code for reuse. No license information was specified in
the provided documentation.
