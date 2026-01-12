![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research--Ready-success)
# 🧠 ModerationDEQ

**Policy-Aware Deep Equilibrium Modeling of Online Community Dynamics**

ModerationDEQ is a machine learning project that models the **long-term equilibrium behavior of online communities under moderation policies** using **Deep Equilibrium Neural Networks (DEQs)**.

Instead of predicting short-term outcomes, this project learns **stable policy-dependent equilibria**, capturing how communities evolve when moderation rules are applied repeatedly over time.

---

## 🎯 Why this project?

Online communities behave like **dynamical systems**:
- moderation creates feedback loops
- effects appear over long time horizons
- short-term predictions are misleading

Most ML models predict **what happens next**.  
**ModerationDEQ answers**:

> *What happens in the long run if a moderation policy stays in place?*

---

## 🧩 Core Idea

We model community evolution as a system:

[z_{t+1} = f(z_t, {policy})]

Instead of unrolling this forever, we directly solve for the **equilibrium**:

[z^* = f(z^*, {policy})]

The equilibrium `z*` represents the **long-term steady state** of the community.

---

## 🔁 System Diagram
Initial Community State (z₀)
│
▼
┌──────────────────────────┐
│ Neural Dynamics Function │
│ f(z, policy) │
└──────────────────────────┘
│
▼
┌──────────────────────────┐
│ Fixed-Point Solver │
│ (Implicit Equilibrium) │
└──────────────────────────┘
│
▼
Long-Term Equilibrium State (z*)

Training uses **implicit differentiation** — gradients flow through the equilibrium, not through time steps.

---

## 🧠 Why Deep Equilibrium Models (DEQs)?

DEQs allow us to:

- ♾️ Model **infinite-depth** behavior
- 🔒 Enforce **stable fixed points**
- 🔁 Capture **feedback loops**
- 🧮 Avoid explicit recurrence
- 📐 Perform **implicit backpropagation**

This makes them ideal for **policy analysis** and **social systems modeling**.

---

## 📊 Community State Representation

Each community is represented by a compact state vector:

| Dimension | Meaning |
|--------|--------|
| 📈 Content Quality | Signal-to-noise ratio |
| ⚠️ Toxicity | Harmful behavior level |
| 🛠 Moderation Pressure | Reports / workload |
| 👥 Engagement | Participation level |

All values are bounded in **[0, 1]** for interpretability.

---

## 🛡 Moderation Policy Parameters

Policies are defined by:

| Parameter | Meaning |
|--------|--------|
| 🔒 Strictness | Aggressiveness of moderation |
| 🎚 Threshold | Tolerance for toxicity |

Policies are embedded **non-linearly** to allow expressive policy effects.

---

## 📈 What the Model Learns

- ✅ Stable equilibria (residuals ~1e-5)
- ✅ Policy-dependent regime shifts
- ✅ Toxicity–engagement trade-offs
- ✅ Robust convergence from different initial states
- ✅ No trivial collapse

This is **system learning**, not prediction.

---

## 🧪 Experiments Included

- Synthetic dataset inspired by real community behavior
- Equilibrium residual tracking
- Policy sweep visualizations
- Multiple initial-state basin tests
- DEQ vs MLP baseline comparison
- Ablation studies (stability vs realism)

---

## ⚙️ Setup & Run

### 1️⃣ Create virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install torch numpy pandas matplotlib
python data/generate_dataset.py
python train.py
python -m analysis.visualize_equilibria
📌 Key Takeaways

❌ Not a classifier

❌ Not a next-step predictor

✅ A long-term equilibrium model

✅ Designed for policy analysis

✅ Uses implicit neural dynamics






