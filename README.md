# 🧭 Landmark-Based Conversational Indoor Navigation

A **mixed-reality indoor navigation system** that generates **human-like, landmark-grounded navigation instructions** from a **single RGB image and a natural-language query**.

This project was developed as part of the **Mixed Reality course (ETH Zurich / University of Zurich)** and focuses on practical, deployable indoor navigation rather than abstract VLN benchmarks.

---

## ✨ Key Features

- 📷 **Single-image user localization** in a reconstructed indoor environment
- 🧭 **Geometric path planning** with collision-free trajectories
- 🏷️ **Semantic landmark extraction** from perceptual observations
- 🗣️ **Concise, human-oriented navigation instructions**
- 🧠 **Lightweight on-device language model** (LoRA fine-tuned)
- 📊 **Quantitative evaluation + user study**

---

## 🛠️ Installation & Setup

Please refer to the [SETUP.md](SETUP.md) for detailed installation instructions.

**Quick Start:**

```bash
git clone https://github.com/rzninvo/CNSG.git
cd CNSG
bash scripts/install.sh
```

---

## 🧠 Motivation

Indoor navigation differs fundamentally from outdoor navigation: GPS is unreliable, maps are incomplete, and humans rely heavily on **local visual landmarks** rather than metric distances.

Most Vision-and-Language Navigation (VLN) systems:

- output low-level actions (_turn left, move forward_), or
- generate abstract, verbose instructions poorly aligned with human intuition.

**Our goal** is to generate navigation instructions the way _people_ naturally do:

> _“Walk past the sofa, then turn right at the stairs.”_

---

## 🏗️ System Overview

**Input**

- A single RGB image captured by the user
- A natural-language navigation query (e.g. _"How do I get to room HG E 3?"_)

**Output**

- Step-by-step navigation instructions grounded in **visible landmarks**

```
User Image + Query
↓
User Localization (2D–3D)
↓
Path Planning (Habitat-Sim)
↓
Landmark Extraction
↓
Instruction Generation (LLM)
```

---

## 📍 User Localization

The user is localized from a **single RGB image** using image-based localization against a pre-built 3D reconstruction:

- Local feature extraction
- Image retrieval
- 2D–3D correspondence matching
- 6-DoF pose estimation (PnP + RANSAC)

The estimated pose initializes the user inside the simulated environment and serves as the start point for navigation.

---

## 🧭 Path Planning

Given the localized user pose and a normalized goal representation, the system computes a **collision-free path** using Habitat-Sim’s planning module.

The resulting trajectory is represented as a sequence of waypoints and acts as the geometric backbone of the navigation pipeline.

---

## 🏷️ Landmark Extraction

To ground instructions in perception:

1. The path is densified into viewpoints
2. At each viewpoint, the simulator provides RGB, depth, and semantic labels
3. Visible objects are clustered into **persistent semantic landmarks**
4. A saliency score filters non-informative objects

This produces a **temporally ordered sequence of landmarks** describing what the user is expected to see along the route.

---

## 🗣️ Instruction Generation

Navigation instructions are generated from:

- the target destination
- the high-level path structure (turns, transitions)
- the ordered landmark observations

The output is **concise, fluent, and spatially grounded**, explicitly referencing visible landmarks and spatial relations.

---

## 🧠 Lightweight Language Model

Instruction generation runs on a **lightweight local language model**, adapted via **LoRA fine-tuning**:

- ✅ No cloud dependency
- ✅ Low GPU memory footprint
- ✅ Suitable for on-device deployment

Despite its small size, the fine-tuned model achieves instruction quality comparable to large proprietary models.

---

## 📊 Experimental Evaluation

The system is evaluated in two settings:

### 🏠 Simulated House Environment

- Known start and goal
- Isolates instruction generation quality

### 🏫 HG Academic Building

- Full pipeline evaluation
- Real image-based localization
- Web-based user interaction

### 📐 Evaluation Metrics

All instructions are rated on a 5-point Likert scale:

- **Reference Object Quality**
- **Spatial & Directional Correctness**
- **Naturalness of Language**

Landmark-grounded pipelines consistently outperform language-only baselines.

---

## ⏱️ Latency Analysis

End-to-end latency remains suitable for interactive use:

- Visual localization is the main bottleneck
- Instruction generation adds minimal overhead
- Local inference ensures stable latency and privacy

---

## 📱 Demo Application

The system is integrated into a **web-based mobile interface** that allows users to:

- Capture an image of their surroundings
- Submit a navigation query
- Receive step-by-step landmark-grounded instructions

The same interface is used for user studies and evaluation.

---

## 🚧 Limitations & Future Work

**Current limitations**

- Single-image localization
- Batch-style interaction

**Planned extensions**

- Real-time egocentric video localization
- Continuous instruction refinement
- Latency optimization
- Deployment on AR glasses

---

## 👥 Contributors

- Riccardo Bianco (ETH Zurich)
- Francesco Bondi (ETH Zurich)
- Roham Zendehdel Nobari (ETH Zurich)
- Shaurya Kishore Panwar (University of Zurich)
- Fatemeh Sadat Daneshmand (ZHAW Winterthur)

**Supervised by**
Mahdi Rad · Gabriele Goletto · Kate Jaroslavceva

---

## 📄 License

MIT License © 2025 Landmark-Based Conversational Indoor Navigation Team
