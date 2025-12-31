# Distributed Construction with Extended Stigmergy (DCES)

**Author:** Adrián Racero Serrano  
**Course:** Multiagent Systems  
**Institution:** Universidad Politécnica de Madrid

## 1. Overview

This project addresses the **Inverse Problem of Construction**: finding a set of low-level local rules that allow a swarm of autonomous robots to build a specific, high-level 2D structure without centralized control or global GPS.

The implementation replicates and analyzes the **extended stigmergy** approach described by Werfel and Nagpal (2006). Unlike standard stigmergy (where actions are driven by the state of the environment), **extended** stigmergy allows environmental elements (building blocks) to store data. In this system, blocks store their relative coordinates, acting as local navigation beacons for the robots.

### Key features
* **Decentralized coordination:** robots operate independently using a Finite State Machine (FSM).
* **Local localization:** robots determine their position relative to the structure by reading data from adjacent blocks (simulating RFID tags).
* **Deadlock prevention:** implements "Algorithm 1" (Werfel & Nagpal) and a "Separation Rule" to guarantee the structure is built without creating unfillable gaps or trapping robots.
* **Scalability analysis:** includes tools to benchmark swarm performance and analyze physical interference using Friedman and Nemenyi statistical tests.

## 2. Repository Structure

* **`DCES.py`**: the core simulation module built with the **Mesa** framework. It contains:
    * `BlockAgent`: passive agents representing building material.
    * `RobotAgent`: active agents that fetch blocks and apply construction rules.
    * `ConstructionModel`: the environment class managing the grid, scheduler, and data collection.
* **`proyect_DCES.ipynb`**: the main entry point. A Jupyter Notebook used to:
    * Configure simulation parameters (target shape, swarm size).
    * Visualize the construction process in real-time.
    * Run systematic experiments for performance analysis.
* **`requirements.txt`**: list of Python dependencies required to run the project.
* **`data/`**: folder containing configuration files for target structures (e.g., binary matrices defining the shape).
* **`figures/`**: output folder for plots and graphs generated during analysis.

## 3. Installation

1.  Clone this repository:
    ```bash
    git clone <your-repository-url>
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies include: `mesa`, `numpy`, `matplotlib`, `pandas`, `seaborn`, `scipy`.*

## 4. Usage

To run the simulation and see the visualization:

1.  Open the Jupyter Notebook:
    ```bash
    jupyter lab proyect_DCES.ipynb
    ```
2.  Run the cells in **Section 3: Simulation Execution**.
    * You can modify `n_robots` to test different swarm sizes.
    * You can change `target_shape` to load different blueprints from the `data/` folder.

To reproduce the **Scalability analysis**:
1.  Run the cells in **Section 4: Performance analysis**.
2.  This will execute a batch run varying the swarm size (e.g., N=1 to N=100) and structure scale, generating boxplots of the "total distance traveled" and "time to completion".