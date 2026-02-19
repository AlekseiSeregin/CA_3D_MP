"""
Test script for diffusion elements (outward and inward).

This script demonstrates:
1. Initializing multiple ActiveElem (outward) and multiple OxidantElem (inward)
2. Using DiffusionEngine with one dedicated pool per element (full parallel: no sequential step)
3. Total workers = (n_outward × OUTWARD_DIFFUSION_WORKERS) + (n_inward × INWARD_DIFFUSION_WORKERS)
   e.g. 2 outward × 4 + 2 inward × 2 = 12 workers in parallel
"""

import numpy as np
from configuration import Config
from configuration.config_utils_classes import ElemInput
from elements.elements import ActiveElem, OxidantElem
from diffusion_3d_mp_example import DiffusionEngine
from utils.utils import Utils


def create_test_settings_outward():
    """Create test settings for outward diffusion element (ActiveElem)."""
    settings = ElemInput()
    settings.ELEMENT = "Cr"
    settings.PROBABILITIES = [0.15, 0.1]  # [p1, p_r_extra] for Chopard-Droz diffusion
    settings.N_PER_PAGE = 10
    settings.CONC_PRECISION = "rand"  # or "exact"
    settings.SPACE_FILL = "full"  # or "half"
    return settings


def create_test_settings_inward():
    """Create test settings for inward diffusion element (OxidantElem)."""
    settings = ElemInput()
    settings.ELEMENT = "O"
    settings.PROBABILITIES = [0.15, 0.1]  # [p1, p_r_extra] for Chopard-Droz diffusion
    settings.PROBABILITIES_2D = 0.5  # Grain boundary diffusion probability
    settings.PROBABILITIES_SCALE = [0.1, 0.2, 0.3]  # Scale diffusion probabilities
    settings.PROBABILITIES_INTERFACE = [0.05, 0.1, 0.15]  # Interface diffusion probabilities
    settings.N_PER_PAGE = 5
    return settings


def count_particles_in_grid(element):
    """Count total particles in the diffusion grid."""
    if not hasattr(element, '_diff_A_count') or element._diff_A_count is None:
        return 0
    
    # Get current read buffer
    read_count = element._diff_A_count if element._diff_read_name == element._diff_shm_A.name else element._diff_B_count
    return int(read_count.sum())


def main():
    print("=" * 70)
    print("Diffusion Elements Test Script")
    print("=" * 70)
    print(f"Grid size: {Config.N_CELLS_PER_AXIS}x{Config.N_CELLS_PER_AXIS}x{Config.N_CELLS_PER_AXIS}")
    print(f"Max particles per cell: {getattr(Config, 'DIFFUSION_MAX_PER_CELL', 50)}")
    n_out = getattr(Config, 'OUTWARD_DIFFUSION_WORKERS', 4)
    n_in = getattr(Config, 'INWARD_DIFFUSION_WORKERS', 2)
    print(f"Outward workers per element: {n_out}")
    print(f"Inward workers per element: {n_in}")
    print(f"Boundary condition: {getattr(Config, 'DIFFUSION_BOUNDARY_X', 'periodic')}")
    print()
    
    # Create Utils instance (needed for OxidantElem)
    utils = Utils()
    
    # Initialize elements: can have multiple outward and multiple inward
    print("Initializing elements...")
    print("-" * 70)
    
    outward_elements = []
    inward_elements = []
    
    # Outward diffusion elements (ActiveElem) - add as many as needed
    for idx, name in enumerate(["Cr", "Al"], start=1):  # example: two outward elements
        s = create_test_settings_outward()
        s.ELEMENT = name
        elem = ActiveElem(s)
        outward_elements.append(elem)
        print(f"✓ ActiveElem (outward) #{idx} {elem.elem_name}: {count_particles_in_grid(elem)} particles")
    
    # Inward diffusion elements (OxidantElem) - add as many as needed
    for idx, name in enumerate(["O", "N"], start=1):  # example: two inward elements (second may be disabled in Config)
        s = create_test_settings_inward()
        s.ELEMENT = name
        try:
            elem = OxidantElem(s, utils)
            inward_elements.append(elem)
            print(f"✓ OxidantElem (inward) #{idx} {elem.elem_name}: {count_particles_in_grid(elem)} particles")
        except Exception as e:
            print(f"  (skipped inward #{idx} {name}: {e})")
    
    # All elements to diffuse: multiple outward + multiple inward (one pool per element)
    elements_to_diffuse = outward_elements + inward_elements
    element_labels = [f"{e.elem_name} (outward)" for e in outward_elements] + [f"{e.elem_name} (inward)" for e in inward_elements]
    
    if not elements_to_diffuse:
        print("No elements to diffuse. Exiting.")
        return
    
    print()
    
    # Create DiffusionEngine: one pool per element (full parallel, no sequential step)
    print("Creating DiffusionEngine (one pool per element for full parallel)...")
    print("-" * 70)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    engine = DiffusionEngine(n_out, n_in, rng)
    total_workers = len(outward_elements) * n_out + len(inward_elements) * n_in
    print(f"✓ Workers: {len(outward_elements)} outward × {n_out} + {len(inward_elements)} inward × {n_in} = {total_workers} in parallel")
    print()
    
    # Apply diffusion steps: all elements diffused in parallel (each with its own pool)
    n_steps = 500
    print(f"Applying {n_steps} diffusion steps (all elements in parallel)...")
    print("-" * 70)
    
    for step in range(n_steps):
        print(f"\nStep {step + 1}/{n_steps}:")
        particles_before = [count_particles_in_grid(e) for e in elements_to_diffuse]
        engine.diffuse_multiple(elements_to_diffuse)
        particles_after = [count_particles_in_grid(e) for e in elements_to_diffuse]
        for label, before, after in zip(element_labels, particles_before, particles_after):
            print(f"  {label}: {before} → {after} particles")
    
    print()
    print("=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    
    # Cleanup
    print("\nCleaning up...")
    engine.close()
    for e in outward_elements + inward_elements:
        e.close_and_unlink_shm()
    print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
