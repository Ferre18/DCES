"""
DCES.py (Distributed Construction with Extended Stigmergy)

This module implements a multi-agent simulation where autonomous robots build 
a user-defined 2D structure. It utilizes the 'Extended Stigmergy' concept 
where blocks store their own relative coordinates, allowing robots to localize 
themselves without a global GPS.

Key components:
1. BlockAgent: passive agents representing building materials.
2. RobotAgent: active agents that fetch blocks and place them based on local rules (Algorithm 1).
3. ConstructionModel: the Mesa environment managing the grid and scheduler.
"""

import mesa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import display, clear_output
import time
import pandas as pd
import seaborn as sns
import scipy.stats as stats

def generate_solid_square(size):
    """
    Generates a simple target matrix representing a solid square.
    
    Args:
        size (int): The width/height of the square.
        
    Returns:
        np.array: A matrix of 1s (structure) of shape (size, size).
    """
    return np.ones((size, size), dtype=int)
    
def load_simulation_config(file_path):
    """
    Parses a custom configuration text file to define the simulation target.
    
    Expected File Format:
    Line 1: GRID_WIDTH <int>
    Line 2: GRID_HEIGHT <int>
    Line 3+: rows of 0s and 1s representing the target shape.
    
    Args:
        file_path (str): path to the config file.
        
    Returns:
        tuple: (width, height, target_matrix_as_numpy_array)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 1. Parse metadata (workspace dimensions)
    # We strip whitespace and split by space to get the values
    width = int(lines[0].strip().split()[1])
    height = int(lines[1].strip().split()[1])
    
    # 2. Parse the structure (the matrix)
    # We take all lines starting from index 2 down to the end
    matrix_lines = lines[2:]
    
    # Convert text lines into a list of lists, then to a NumPy array
    matrix_data = []
    for line in matrix_lines:
        # Removes whitespace, splits by space, converts '1'/'0' to integers
        row = [int(x) for x in line.strip().split()]
        if row:  # Only add non-empty rows
            matrix_data.append(row)
            
    target_matrix = np.array(matrix_data)
    
    return width, height, target_matrix

def plot_target_shape(target_matrix):
    """
    Visualizes the desired structure matrix before simulation starts.
    Calculates the geometric center to identify which block acts as the 'Seed'.
    
    Args:
        target_matrix (np.array): binary matrix of the structure.
    """
    rows, cols = target_matrix.shape
    
    # --- 1. Identify the seed ---
    # We find the block closest to the geometric center of this matrix
    valid_indices = np.argwhere(target_matrix == 1)
    
    if len(valid_indices) > 0:
        center_y, center_x = rows // 2, cols // 2
        dists = np.sqrt((valid_indices[:,0] - center_y)**2 + 
                        (valid_indices[:,1] - center_x)**2)
        seed_pos = valid_indices[np.argmin(dists)]
    else:
        seed_pos = None

    # --- 2. Prepare visualization grid ---
    # We copy the matrix so we don't modify the original
    plot_data = target_matrix.copy()
    
    # Mark the seed with a unique value (5) for coloring
    if seed_pos is not None:
        plot_data[seed_pos[0], seed_pos[1]] = 5

    # --- 3. Plotting ---
    # Colormap: 0=White, 1=Silver, 5=Dark Gray
    cmap = colors.ListedColormap(['white', 'silver', '#404040'])
    bounds = [-0.5, 0.5, 1.5, 5.5] # Buckets for 0, 1, and 5
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(5, 5))
    plt.imshow(plot_data, cmap=cmap, norm=norm, origin='upper')

    # Add grid lines to see the cells clearly
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    plt.xticks(np.arange(-0.5, cols, 1))
    plt.yticks(np.arange(-0.5, rows, 1))
    
    # Hide tick labels for a cleaner look
    plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.title(f"Target shape: {cols}x{rows}")
    plt.show()

class BlockAgent(mesa.Agent):
    """
    A passive agent representing a building block.
    Can be in three states:
    1. Loose (available for pickup).
    2. Structure (part of the built wall).
    3. Seed (the initial reference point).
    """
    def __init__(self, model, rel_x, rel_y, is_structure=False, is_seed=False):
        super().__init__(model)
        self.rel_x = rel_x
        self.rel_y = rel_y
        self.is_structure = is_structure 
        self.is_seed = is_seed
    def step(self): pass

class RobotAgent(mesa.Agent):
    """
    The active builder robot. 
    Implements a Finite State Machine:
    1. Search for loose block (Fetch).
    2. Move to structure (Home).
    3. Navigate perimeter (Wall Follow).
    4. Check construction rules (Algorithm 1).
    5. Deposit block.
    """
    def __init__(self, model):
        super().__init__(model)
        self.carrying_block = False 
        self.seen_row_start = False 
        self.last_pos = None 
        self.heading = self.random_heading() 
        self.VIEW_RADIUS = 3                 

    def random_heading(self):
        """Pick a random cardinal direction (N, S, E, W)."""
        return self.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    # --- SENSORS & HELPERS ---
        
    def get_valid_moves(self):
        """Returns neighbor coordinates that do not contain obstacles (Blocks OR Robots)."""
        if not self.pos: return []
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        
        valid = []
        for p in possible_steps:
            contents = self.model.grid.get_cell_list_contents(p)
            
            # Check for ANY obstacle: BlockAgent OR RobotAgent
            has_obstacle = any(isinstance(obj, (BlockAgent, RobotAgent)) for obj in contents)
            
            if not has_obstacle:
                valid.append(p)
                
        return valid

    def is_structure(self, pos):
        """Checks if a specific grid position contains a structural block."""
        if self.model.grid.out_of_bounds(pos): return False
        content = self.model.grid.get_cell_list_contents(pos)
        return any(isinstance(o, BlockAgent) and o.is_structure for o in content)

    def get_structure_neighbors(self, moore=True):
        """Returns list of adjacent positions that are part of the structure."""
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=moore, include_center=False)
        return [p for p in neighbors if self.is_structure(p)]

    def grid_to_matrix(self, pos):
        """
        Extended Stigmergy Core:
        Converts global absolute grid coordinates (Environment) into 
        relative matrix coordinates (Blueprint) based on the shift from the Seed.
        """
        gx, gy = pos
        # GridPos = MatrixPos + Shift  ->  MatrixPos = GridPos - Shift
        mx = gx - self.model.shift_x
        my = gy - self.model.shift_y
        rows, cols = self.model.target_matrix.shape
        if 0 <= mx < cols and 0 <= my < rows:
            return int(mx), int(my)
        return None, None

    # --- MOVEMENT HELPERS ---
    
    def move_towards(self, target_pos, noise=0.1):
        """Moves one step closer to target_pos (Manhattan distance) with slight noise."""
        valid_moves = self.get_valid_moves()
        if not valid_moves: return
        
        if self.random.random() < noise:
            self.model.grid.move_agent(self, self.random.choice(valid_moves))
        else:
            valid_moves.sort(key=lambda p: abs(p[0] - target_pos[0]) + abs(p[1] - target_pos[1]))
            self.model.grid.move_agent(self, valid_moves[0])

    def move_straight_with_bounce(self):
        """
        Movement for searching (random walk with persistence).
        Moves straight until it hits a wall/block, then picks a new random direction.
        """
        dx, dy = self.heading
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        target_pos = (new_x, new_y)
        
        # Check if target has a block
        if not self.model.grid.out_of_bounds(target_pos):
            contents = self.model.grid.get_cell_list_contents(target_pos)
            if any(isinstance(o, BlockAgent) for o in contents):
                # Hit a block? Bounce.
                self.heading = self.random_heading()
                return

        # Move if valid
        if (not self.model.grid.out_of_bounds(target_pos) and 
            self.model.grid.is_cell_empty(target_pos)):
            self.model.grid.move_agent(self, target_pos)
        else:
            self.heading = self.random_heading()

    def scan_for_local_block(self):
        """
        Scans local radius for loose blocks to pick up.
        Returns closest block position or None.
        """
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.VIEW_RADIUS
        )
        loose_blocks = [
            obj for obj in neighbors 
            if isinstance(obj, BlockAgent) and not obj.is_structure
        ]
        if not loose_blocks:
            return None
        best_block = min(loose_blocks, key=lambda b: 
                         abs(b.pos[0] - self.pos[0]) + abs(b.pos[1] - self.pos[1]))
        return best_block.pos

    # --- SAFETY CHECK ---
    
    def check_separation_violation(self, pos):
        """
        Implements the 'Separation Rule' to prevent unfillable gaps.
        Ensures placing a block at 'pos' doesn't create a 1-block gap 
        between the new block and existing structure.
        """
        x, y = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            dist = 1
            gap_detected = False
            while True:
                nx, ny = x + (dx * dist), y + (dy * dist)
                if not (0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height): break
                is_struct = self.is_structure((nx, ny))
                mx, my = self.grid_to_matrix((nx, ny))
                should_be_structure = False
                if mx is not None: should_be_structure = (self.model.target_matrix[my, mx] == 1)
                
                if is_struct:
                    if gap_detected: return True 
                    else: break 
                else:
                    if should_be_structure: gap_detected = True
                    else: break
                dist += 1
        return False

    # --- PERIMETER FOLLOWING ---
    
    def get_perimeter_move(self):
        """
        Calculates the next move to traverse the structure perimeter 
        in a counter-clockwise direction.
        """
        if not self.pos: return None
        current_x, current_y = self.pos
        valid_moves = self.get_valid_moves()
        if not valid_moves: return None
        
        perimeter_moves = []
        for mv in valid_moves:
            neighbors_of_move = self.model.grid.get_neighborhood(mv, moore=True, include_center=False)
            if any(self.is_structure(n) for n in neighbors_of_move):
                perimeter_moves.append(mv)
        
        if not perimeter_moves: return self.random.choice(valid_moves)

        if self.last_pos in perimeter_moves and len(perimeter_moves) > 1:
            perimeter_moves.remove(self.last_pos)

        struct_neighbors = self.get_structure_neighbors(moore=True)
        if not struct_neighbors: return perimeter_moves[0]
        
        avg_sx = sum(p[0] for p in struct_neighbors) / len(struct_neighbors)
        avg_sy = sum(p[1] for p in struct_neighbors) / len(struct_neighbors)
        
        vec_x = current_x - avg_sx
        vec_y = current_y - avg_sy
        target_vec_x = -vec_y
        target_vec_y = vec_x
        
        best_move = max(perimeter_moves, key=lambda m: 
                        (m[0]-current_x)*target_vec_x + (m[1]-current_y)*target_vec_y)
        return best_move

    # --- ALGORITHM 1 CHECKS ---

    def is_inside_corner(self):
        """Checks if the current position is an 'inside corner' (surrounded by >2 blocks)."""
        struct_neighbors = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        count = sum(1 for p in struct_neighbors if self.is_structure(p))
        return count >= 2

    def is_end_of_row(self, next_pos):
        """
        Checks if the position is the end of a row.
        Defined as having only 1 structural neighbor AND the next spot in the traversal 
        points to an empty space in the blueprint.
        """
        if next_pos:
            mx, my = self.grid_to_matrix(next_pos)
            if mx is not None and self.model.target_matrix[my, mx] == 0:
                return True
        struct_neighbors = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        count = sum(1 for p in struct_neighbors if self.is_structure(p))
        return count == 1

    # --- MAIN MOVE LOGIC ---
    
    def move(self):
        # 0. ANTI-STACKING CHECK
        # If we are standing on top of a structure block, we must NOT build and should move away.
        if self.is_structure(self.pos):
            # Move to a random neighbor to get off the wall
            valid_moves = self.get_valid_moves()
            if valid_moves:
                self.model.grid.move_agent(self, self.random.choice(valid_moves))
            return

        # 1. FETCH STATE
        if not self.carrying_block:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)
            for agent in neighbors:
                if isinstance(agent, BlockAgent) and not agent.is_structure:
                    self.model.grid.remove_agent(agent)
                    self.carrying_block = True
                    self.seen_row_start = False 
                    self.last_pos = None       
                    return

            target_block_pos = self.scan_for_local_block()
            if target_block_pos:
                self.move_towards(target_block_pos)
            else:
                self.move_straight_with_bounce()
            return

        # 2. TRAVEL TO SEED
        if not self.get_structure_neighbors(moore=True):
            seed_agents = [a for a in self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=50) 
                           if isinstance(a, BlockAgent) and a.is_seed]
            
            if seed_agents:
                target = seed_agents[0].pos
            else:
                # Fallback: go to any structural block
                structs = [a for a in self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=50) 
                           if isinstance(a, BlockAgent) and a.is_structure]
                if structs:
                    target = structs[0].pos
                else:
                    target = (self.model.grid.width // 2, self.model.grid.height // 2)
            
            self.move_towards(target)
            return

        # 3. BUILD
        mx, my = self.grid_to_matrix(self.pos)
        should_occupy = False
        if mx is not None:
             should_occupy = (self.model.target_matrix[my, mx] == 1)
        
        next_pos = self.get_perimeter_move()
        at_end_of_row = self.is_end_of_row(next_pos)
        at_inside_corner = self.is_inside_corner()

        alg1_allowed = should_occupy and (at_inside_corner or (self.seen_row_start and at_end_of_row))
        
        safety_check_passed = True
        if alg1_allowed:
            if self.check_separation_violation(self.pos):
                safety_check_passed = False
        
        # FINAL CHECK: ensure we are not building on top of an existing block
        # (Double check, although Step 0 handles this for 'self.pos', logic must prevent overwrite)
        can_build = alg1_allowed and safety_check_passed and not self.is_structure(self.pos)

        if can_build:
            new_block = BlockAgent(self.model, 0, 0, is_structure=True)
            self.model.grid.place_agent(new_block, self.pos)
            self.model.num_blocks_placed += 1
            self.carrying_block = False
            self.last_pos = None 
            if next_pos and self.model.grid.is_cell_empty(next_pos):
                self.model.grid.move_agent(self, next_pos)
        else:
            if at_end_of_row:
                self.seen_row_start = True
            if next_pos:
                self.last_pos = self.pos 
                self.model.grid.move_agent(self, next_pos)

    def step(self):
        if not self.pos: return
        self.move()

class ConstructionModel(mesa.Model):
    """
    The environment controller.
    1. Initializes the grid.
    2. Spawns the seed.
    3. Spawns robots.
    4. Spawns loose blocks.
    """
    def __init__(self, target_matrix, N_robots=5, width=None, height=None):
        super().__init__()
        self.target_matrix = target_matrix
        self.running = True
        self.num_blocks_placed = 0

        self.total_structure_blocks = int(np.sum(target_matrix)) - 1
        
        # --- 1. Dynamic grid sizing ---
        rows, cols = target_matrix.shape
        total_structure_blocks = int(np.sum(target_matrix))
        total_loose_blocks = total_structure_blocks + 20
        total_agents = N_robots + total_loose_blocks
        
        if width is None or height is None:
            min_dim_struct = max(rows, cols) + 12 
            TARGET_DENSITY = 0.25 
            required_area = total_agents / TARGET_DENSITY
            min_dim_density = int(np.sqrt(required_area))
            optimal_size = max(min_dim_struct, min_dim_density)
            if optimal_size % 2 == 0: optimal_size += 1
            width = optimal_size
            height = optimal_size
            
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        
        # --- 2. Seed setup & coordinate mapping ---
        # Identify the seed in the matrix
        valid_indices = np.argwhere(target_matrix == 1)
        matrix_center_y, matrix_center_x = rows // 2, cols // 2
        dists = np.sqrt((valid_indices[:,0] - matrix_center_y)**2 + 
                        (valid_indices[:,1] - matrix_center_x)**2)
        self.seed_matrix_pos = valid_indices[np.argmin(dists)]
        seed_r, seed_c = self.seed_matrix_pos

        # Calculate the shift: we want matrix center to align with grid center
        grid_center_y, grid_center_x = height // 2, width // 2
        
        # SHIFT = Grid_Center - Matrix_Center
        self.shift_x = grid_center_x - matrix_center_x
        self.shift_y = grid_center_y - matrix_center_y
        
        # Calculate where the seed lands using this shift
        final_seed_x = seed_c + self.shift_x
        final_seed_y = seed_r + self.shift_y
        
        # Place Seed
        seed_block = BlockAgent(self, 0, 0, is_structure=True, is_seed=True)
        self.grid.place_agent(seed_block, (final_seed_x, final_seed_y))
        
        # --- 3. Place robots ---
        start_positions = [(0,0), (0, height-1), (width-1, 0), (width-1, height-1)]
        for i in range(N_robots):
            pos = start_positions[i % 4]
            robot = RobotAgent(self)
            self.grid.place_agent(robot, pos)
            
        # --- 4. Spawn loose blocks ---
        placed_count = 0
        while placed_count < total_loose_blocks:
            rx = self.random.randrange(width)
            ry = self.random.randrange(height)
            # Spawn away from the actual grid center
            if self.grid.is_cell_empty((rx, ry)) and (abs(rx - grid_center_x) > 5 or abs(ry - grid_center_y) > 5):
                self.grid.place_agent(BlockAgent(self, 0, 0), (rx, ry))
                placed_count += 1
    
    def step(self):
        self.agents.shuffle_do("step")

        if self.num_blocks_placed >= self.total_structure_blocks:
            self.running = False

def run_simulation(target_shape, N_robots=5, width=None, height=None, max_steps=1000, delay=0.001):
    """
    Runs the construction simulation with real-time visualization in Jupyter.
    
    Args:
        target_shape (np.array): the desired structure.
        N_robots (int): number of active agents.
        max_steps (int): safety limit to stop simulation.
        delay (float): time in seconds between frames (animation speed).
    """
    
    # 1. Initialize model
    model = ConstructionModel(target_shape, N_robots=N_robots, width=width, height=height)
    
    # Calculate goal dynamically
    total_blocks_needed = np.sum(target_shape) - 1 
    
    print(f"Starting Simulation... Goal: {total_blocks_needed} blocks")
    
    for i in range(max_steps):
        model.step()
        
        # --- Visualization Logic ---
        clear_output(wait=True)
        
        # Extract grid data for plotting
        grid_data = np.zeros((model.grid.height, model.grid.width))
        
        for content, (x, y) in model.grid.coord_iter():
            for obj in content:
                if isinstance(obj, BlockAgent):
                    if obj.is_structure:
                        # 5 = Seed, 1 = Structure
                        grid_data[y][x] = 5 if obj.is_seed else 1
                    else:
                        grid_data[y][x] = 3 # Loose block
                elif isinstance(obj, RobotAgent):
                    # 2 = Robot carrying, 4 = Robot empty
                    grid_data[y][x] = 2 if obj.carrying_block else 4

        # Colormap: 
        # 0: white, 1: silver, 2: red, 3: blue, 4: green, 5: dimgray
        cmap = colors.ListedColormap(['white', 'silver', 'red', 'blue', 'green', '#404040'])
        
        plt.figure(figsize=(6, 6))
        plt.imshow(grid_data, cmap=cmap, vmin=0, vmax=5, origin='upper')
        
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(-0.5, model.grid.width, 1))
        plt.yticks(np.arange(-0.5, model.grid.height, 1))
        plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        plt.title(f"Step: {i+1} | Built: {model.num_blocks_placed}/{total_blocks_needed}")
        plt.show()
        
        # Stop condition
        if model.num_blocks_placed >= total_blocks_needed:
            print("Construction Completed!")
            break
            
        time.sleep(delay)

def benchmark_single_structure(target_shape, swarm_sizes, trials_per_size=20, max_steps=5000, width=None, height=None):
    """
    Runs invisible simulations to measure performance vs swarm size.
    Returns a pandas DataFrame of the results.
    """
    results = []
    total_blocks_to_build = np.sum(target_shape) - 1 

    print(f"--- Starting scalability analysis ---")
    print(f"Goal: build {total_blocks_to_build} blocks.")

    for n_robots in swarm_sizes:
        print(f"Testing swarm size: {n_robots}...", end=" ")
        
        for trial in range(trials_per_size):
            # Pass dynamic dimensions
            model = ConstructionModel(target_shape, N_robots=n_robots, width=width, height=height)
            
            steps = 0
            while model.running and steps < max_steps:
                model.step()
                steps += 1
            
            # STORE RAW DATA (Not Averages)
            results.append({
                "Swarm Size": n_robots,
                "Trial": trial,
                "Time (Steps)": steps
            })
        print("Done.")

    return pd.DataFrame(results)

def benchmark_multi_scale(structure_scales, swarm_sizes, trials_per_config=10, max_steps=5000, width=None, height=None):
    """
    Advanced benchmarking: varies both structure size (scale) and swarm size.
    Calculates metrics like total distance and steps per block.
    """
    results = []
    
    print(f"Starting systematic experiment...")
    print(f"Scales to test: {structure_scales}")
    print(f"Swarm sizes to test: {swarm_sizes}")
    
    for scale in structure_scales:
        # Create a dynamic target for this scale
        target_matrix = generate_solid_square(scale)
        total_blocks = np.sum(target_matrix) - 1
        
        for n_robots in swarm_sizes:
            print(f"  > Testing scale {scale}x{scale} with {n_robots} robots ({trials_per_config} trials)...")
            
            for trial in range(trials_per_config):
                # Initialize your existing model
                model = ConstructionModel(target_matrix, N_robots=n_robots, width=None, height=None)
                
                steps = 0
                while model.running and steps < max_steps:
                    model.step()
                    steps += 1
                
                # --- METRICS CALCULATION ---
                # 1. Construction time: How long did it take?
                time_taken = steps
                
                # 2. Total distance traveled (The "Quality" metric from the paper)
                #    Distance = Time * Num_Robots (assuming 1 step = 1 unit distance)
                total_distance = steps * n_robots
                
                # 3. Efficiency: Steps per Block
                #    Lower is better.
                steps_per_block = time_taken / total_blocks
                
                results.append({
                    "Scale": scale,
                    "Swarm Size": n_robots,
                    "Trial_ID": trial,
                    "Time (Steps)": time_taken,
                    "Total Distance": total_distance,
                    "Steps Per Block": steps_per_block,
                    "Completion": (model.num_blocks_placed / total_blocks) * 100
                })

    return pd.DataFrame(results)