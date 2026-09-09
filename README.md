# Autonomous Intersection Navigation via Curriculum RL & Vision-Language Models (VLM) in CARLA

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CARLA Simulator](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)
[![RL Algorithm](https://img.shields.io/badge/RL-Soft_Actor--Critic_(SAC)-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end autonomous driving research platform designed to resolve complex, occluded, and unsignalized **Right-of-Way (RoW)** urban intersections in the CARLA simulator. The agent is trained through a structured **5-phase Curriculum Learning** pipeline combining Soft Actor-Critic (SAC) reinforcement learning with a multi-modal Vision-Language Model (VLM) decision pipeline.

---

## 🚀 Key Highlights

* **5-Phase Curriculum Pipeline:** Progresses from elementary lateral control to handling aggressive, multi-agent intersection interactions.
* **8 Benchmark RoW Scenarios:** Covers emergency yield amushes, double cross-traffic, slow convoys, aggressive multi-vehicle insertions, partial-view occluders, crosswalk pedestrians, and cyclist hazards.
* **Decoupled Architecture:** Utilizes ZeroMQ (ZMQ) for fast, robust inter-process communication between CARLA synchronous physics stepping and the RL observation pipeline.
* **VLM Zero-Shot Reasoning:** Integrates multimodal semantic scene reasoning for handling critical edge-case blind spots and priority assessment.

---

## 📊 Curriculum Learning Framework

Rather than exposing the policy to complex intersection interactions immediately, training follows a progressive curriculum:

| Phase | Directory | Training Objective | Convergence Result |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `phase1/` | Base lane keeping, speed tracking & error minimization (no obstacles). | `convergence_plot_phase1.png` |
| **Phase 2** | `phase2/` | Longitudinal radar integration & collision avoidance with static obstacles. | `convergence_plot_phase2.png` |
| **Phase 3** | `phase3/` | Adaptive cruising & headway distance management with dynamic lead vehicles. | `convergence_plot_phase3.png` |
| **Phase 4** | `phase4/` | Multi-scenario Right-of-Way priority reasoning in unsignalized junctions. | `convergence_plot_phase4.png` |
| **Phase 5** | `phase5/` | Multimodal VLM semantic reasoning for occluded threats and edge cases. | Complete Integration |

---

## 🚦 Benchmark Scenarios (Phase 4 & 5)

The scenario generator (`carla_sync_v1.py`) deterministically benchmarks the agent across 8 real-world intersection conflicts:

* **Scenario 1:** Emergency Vehicle Ambush (High-speed crossing priority yield)
* **Scenario 2:** Double Cross Traffic (Two sequential lateral cross-threats)
* **Scenario 3:** Slow Convoy Lead + Lateral Cross Threat
* **Scenario 4:** Aggressive Multi-Vehicle Gap Acceptance
* **Scenario 5:** Partial-View Static Occluder with Hidden Crossing Vehicle
* **Scenario 6:** Unregulated Pedestrian Crosswalk Crossing
* **Scenario 7:** In-Path Cyclist Obstacle with Lateral Pedestrian Crossing
* **Scenario 8:** Real Crosswalk Pedestrian + Synchronized Intersection Traffic Combo

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* **CARLA Simulator** (v0.9.13 or newer)
* **Python** 3.8 to 3.12
* NVIDIA GPU with CUDA support

### 2. Clone Repository
```bash
git clone [https://github.com/arash-akhlaghi/carla-rl-vlm-driving.git](https://github.com/arash-akhlaghi/carla-rl-vlm-driving.git)
cd carla-rl-vlm-driving
