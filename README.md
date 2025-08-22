<img width="1211" height="668" alt="image" src="https://github.com/user-attachments/assets/2b93181b-e678-4173-ba2b-73f39dfa5d75" />
# DynaMoE

This script runs inference using Deca’s dynamic mixture-of-experts architecture, powered by DynAMoE Router and DynAMoE Weaver. It supports both static and dynamic routing across multiple expert modules, enabling reproducible, cost-aware inference.
🔧 Features
- Supports Router (single-pass expert selection) and Weaver (multi-hop expert chaining)
- Modular expert loading (MoE1, MoE2, MoE3, etc.)
- Configurable routing logic and fallback strategies
- Reproducible outputs with seed control
- Lightweight CLI interface for batch or single input
