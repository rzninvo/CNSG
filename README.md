# Conversational Indoor Navigation (Mixed Reality Project, ETH/UZH)

## 🧭 Overview

This project aims to develop a **conversational, landmark-based navigation assistant** for **smart glasses**.  
Unlike traditional Vision-and-Language Navigation (VLN) systems that output low-level actions (turn left, go forward),  
our goal is to generate **natural, human-like verbal guidance** grounded in 3D indoor environments such as those from the **Replica** and **Matterport3D** datasets.

The assistant should describe paths using **recognizable landmarks** ("walk past the sofa and enter the kitchen") and support **interactive dialogue**,  
allowing users to ask for clarifications when instructions are unclear (“I don’t see the red chair—can you guide me differently?”).

Ultimately, this system will serve as a prototype for **smart-glass–based navigation**—bridging spatial reasoning, natural language, and 3D perception.

---

## 🧩 Project Objectives

- Build a **Habitat-based VLN environment** that supports arbitrary start–goal navigation inside indoor scans.  
- Extract **landmark-rich routes** between two points in a 3D scene.  
- Generate **LLM-based natural language instructions** describing these routes.  
- Develop a **conversational layer** to handle user feedback and rephrase instructions dynamically.  
- Integrate results into a **demo app** showing both 3D visualization (avatar perspective) and voice-based interaction.

---

## 🧱 Current Pipeline

### 1. Scene Loading (Habitat-Sim / Habitat-Lab)
- Load indoor environments from **Replica** or **Matterport3D** datasets.
- Generate a **navigable mesh** and enable semantic rendering.
- Access 3D scene information: RGB, depth, and semantic labels.

### 2. Path Generation
- Given arbitrary **start** and **goal** coordinates, compute the **shortest path** using Habitat’s `ShortestPath` API.
- Optionally sample intermediate waypoints for route segmentation.

### 3. Landmark Extraction
- Query semantic annotations around each waypoint.
- Identify nearby objects, room types, or architectural features.
- Build a **landmark graph** representing rooms, connections, and salient visual anchors.

### 4. Route Graph Representation
- Encode the route as a graph:
  ```json
  {
    "nodes": [
      {"id":0, "room":"living room", "landmarks":["sofa","lamp"]},
      {"id":1, "room":"hallway", "landmarks":["painting","door"]},
      {"id":2, "room":"kitchen", "landmarks":["fridge","table"]}
    ],
    "edges": [[0,1],[1,2]]
  }

* This serves as the input context for the language model.

### 5. Language Instruction Generation

* Translate the route graph into **natural, landmark-centric navigation instructions** using an LLM.
* Two generation modes:

  1. **Template-based:** rule-based phrasing (“Walk past the [landmark] and enter the [room]”).
  2. **Prompted LLM-based:** a large language model (e.g., GPT, Mistral) conditioned on the route graph.

### 6. Conversational Layer (Interactive Dialogue)

* Maintain conversational memory.

* Allow user clarifications:

  > “I don’t see the painting.”
  > “Okay, after the sofa, look for the white door on your right.”

* The LLM adapts phrasing and focus dynamically using spatial context.

### 7. Visualization & Demo

* Visualize navigation path and avatar movement in **Unity** or **Unreal Engine**.
* Connect to iPad or AR glasses for voice input/output.

---

## 🔍 Technical Architecture

```
┌──────────────────────────────┐
│  Scene Loader (Habitat)      │
│  - Replica / Matterport3D    │
└─────────────┬────────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  Path & Landmark Module  │
   │  - A* planner            │
   │  - Semantic extraction   │
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  Route Graph Builder     │
   │  - Nodes: landmarks      │
   │  - Edges: connections    │
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  LLM Instruction Engine  │
   │  - Template / LLM prompts│
   │  - Conversational loop   │
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  Visualization / Demo    │
   │  - Avatar / 3D path      │
   │  - Speech interface      │
   └──────────────────────────┘
```

---

## 🧠 Related Work & References

Our design builds on several key works in Vision-and-Language Navigation and embodied instruction generation:

| Paper                                                                                      | Description                                                                                                       |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **[VLN-CE (Krantz et al.)](https://github.com/jacobkrantz/VLN-CE)**                        | Habitat-based continuous VLN benchmark. Provides environment setup, navigation API, and pretrained baselines.     |
| **[NaVid-VLN (Zhang et al., RSS 2024)](https://github.com/jzhzhang/NaVid-VLN-CE)**         | Video-based VLN agent that predicts next steps from egocentric view; useful for integrating smart-glasses vision. |
| **[StreamVLN (InternRobotics, 2025)](https://github.com/InternRobotics/StreamVLN)**        | Streaming vision-language navigation framework for continuous video input and real-world deployment.              |
| **[Habitat-Sim / Habitat-Lab (Meta AI)](https://github.com/facebookresearch/habitat-sim)** | Simulator and API for embodied navigation, supporting Replica and Matterport3D datasets.                          |
| **[Replica Dataset](https://github.com/facebookresearch/Replica-Dataset)**                 | High-fidelity indoor scenes with semantic annotations used for navigation and spatial reasoning.                  |

---

## 🚧 Planned Extensions

* **LoRA fine-tuning** of the instruction LLM on landmark-rich navigation data (WP3).
* **Speech interface** integration for voice queries and responses.
* **Comparative evaluation** between rule-based, LLM-based, and hybrid instruction systems.
* **User study** on clarity and spatial alignment of generated navigation cues.

---

## 🧑‍💻 Contributors

* **Riccardo Bianco**
* **Francesco Bondi**
* **Roham Zendehdel Nobari**
* **Shaurya Kishore Panwar**
* **Fatemeh Sadat Daneshmand**

Supervised by: *Mahdi Rad, Gabriele Goletto, and Kate Jaroslavceva*
Mixed Reality Project – ETH Zurich / University of Zurich (Fall 2025)

---

## 📄 License

MIT License © 2025 Conversational Indoor Navigation Team
