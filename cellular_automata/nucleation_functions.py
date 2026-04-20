from utils.numba_functions import *
from utils.numba_functions import (
    nucleation_subblock_kernel_owner,
    nucleation_subblock_kernel_simple_owner,
    nucleation_subblock_kernel_stoich_owner,
    nucleation_subblock_kernel_simple_stoich_owner,
    nucleation_subblock_kernel_owner_spec,
    nucleation_subblock_kernel_simple_owner_spec,
    nucleation_subblock_kernel_stoich_owner_spec,
    nucleation_subblock_kernel_simple_stoich_owner_spec,
    nucleation_subblock_kernel_owner_fold,
    nucleation_subblock_kernel_stoich_owner_fold,
    nucleation_subblock_kernel_owner_spec_fold,
    nucleation_subblock_kernel_stoich_owner_spec_fold,
    nucleation_subblock_kernel_owner_blockmask,
    nucleation_subblock_kernel_simple_owner_blockmask,
    nucleation_subblock_kernel_stoich_owner_blockmask,
    nucleation_subblock_kernel_simple_stoich_owner_blockmask,
    nucleation_subblock_kernel_owner_spec_blockmask,
    nucleation_subblock_kernel_simple_owner_spec_blockmask,
    nucleation_subblock_kernel_stoich_owner_spec_blockmask,
    nucleation_subblock_kernel_simple_stoich_owner_spec_blockmask,
    nucleation_subblock_kernel_owner_fold_blockmask,
    nucleation_subblock_kernel_stoich_owner_fold_blockmask,
    nucleation_subblock_kernel_owner_spec_fold_blockmask,
    nucleation_subblock_kernel_stoich_owner_spec_fold_blockmask,
)
from multiprocessing import shared_memory
from .neigh_indexes import *


def _run_nucleation_legacy_simple_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_simple_owner(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        n_cells,
        seed,
    )


def _run_nucleation_legacy_simple_owner_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_simple_owner_blockmask(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_legacy_prob_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_owner(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_legacy_prob_owner_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_owner_blockmask(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_legacy_prob_owner_fold(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_owner_fold(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_legacy_prob_owner_fold_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_owner_fold_blockmask(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_stoich_simple_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_simple_stoich_owner(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        threshold_outward,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        n_cells,
        seed,
    )


def _run_nucleation_stoich_simple_owner_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_simple_stoich_owner_blockmask(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        threshold_outward,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_stoich_prob_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_stoich_owner(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        threshold_outward,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_stoich_prob_owner_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_stoich_owner_blockmask(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        threshold_outward,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_stoich_prob_owner_fold(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_stoich_owner_fold(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        threshold_outward,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_stoich_prob_owner_fold_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_stoich_owner_fold_blockmask(
        oxidant,
        oxidant_dirs,
        active,
        active_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        threshold_outward,
        seed_slab_k,
        plane_indexes,
        active_check_offsets,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_legacy_simple_owner_spec(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_simple_owner_spec(
        oxidant,
        oxidant_dirs,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        n_cells,
        seed,
    )


def _run_nucleation_legacy_simple_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_simple_owner_spec_blockmask(
        oxidant,
        oxidant_dirs,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_legacy_prob_owner_spec(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_owner_spec(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_legacy_prob_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_owner_spec_blockmask(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_legacy_prob_owner_spec_fold(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_owner_spec_fold(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_legacy_prob_owner_spec_fold_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_owner_spec_fold_blockmask(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_stoich_simple_owner_spec(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_simple_stoich_owner_spec(
        oxidant,
        oxidant_dirs,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        seed_slab_k,
        plane_indexes,
        n_cells,
        seed,
    )


def _run_nucleation_stoich_simple_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_simple_stoich_owner_spec_blockmask(
        oxidant,
        oxidant_dirs,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        seed_slab_k,
        plane_indexes,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_stoich_prob_owner_spec(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_stoich_owner_spec(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_stoich_prob_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_stoich_owner_spec_blockmask(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


def _run_nucleation_stoich_prob_owner_spec_fold(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
):
    nucleation_subblock_kernel_stoich_owner_spec_fold(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
    )


def _run_nucleation_stoich_prob_owner_spec_fold_blockmask(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product,
    full_3d,
    product_init,
    product_x_nzs,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    product_state,
    phase_id,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    nucleation_subblock_kernel_stoich_owner_spec_fold_blockmask(
        oxidant,
        oxidant_dirs,
        product_init,
        product_state,
        phase_id,
        ox_num,
        threshold_inward,
        seed_slab_k,
        plane_indexes,
        flat_neigh_offsets,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        seed,
        block_mask_bits,
        block_cells_x,
        block_cells_y,
        block_cells_z,
    )


_NUCLEATION_KERNEL_RUNNERS = {
    "legacy_simple_owner": _run_nucleation_legacy_simple_owner,
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner,
    "stoich_simple_owner": _run_nucleation_stoich_simple_owner,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner,
}

_NUCLEATION_KERNEL_RUNNERS_SPEC = {
    "legacy_simple_owner": _run_nucleation_legacy_simple_owner_spec,
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_spec,
    "stoich_simple_owner": _run_nucleation_stoich_simple_owner_spec,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_spec,
}

_NUCLEATION_KERNEL_RUNNERS_FOLD = {
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_fold,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_fold,
}

_NUCLEATION_KERNEL_RUNNERS_SPEC_FOLD = {
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_spec_fold,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_spec_fold,
}

_NUCLEATION_KERNEL_RUNNERS_BLOCKMASK = {
    "legacy_simple_owner": _run_nucleation_legacy_simple_owner_blockmask,
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_blockmask,
    "stoich_simple_owner": _run_nucleation_stoich_simple_owner_blockmask,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_blockmask,
}

_NUCLEATION_KERNEL_RUNNERS_SPEC_BLOCKMASK = {
    "legacy_simple_owner": _run_nucleation_legacy_simple_owner_spec_blockmask,
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_spec_blockmask,
    "stoich_simple_owner": _run_nucleation_stoich_simple_owner_spec_blockmask,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_spec_blockmask,
}

_NUCLEATION_KERNEL_RUNNERS_FOLD_BLOCKMASK = {
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_fold_blockmask,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_fold_blockmask,
}

_NUCLEATION_KERNEL_RUNNERS_SPEC_FOLD_BLOCKMASK = {
    "legacy_prob_owner": _run_nucleation_legacy_prob_owner_spec_fold_blockmask,
    "stoich_prob_owner": _run_nucleation_stoich_prob_owner_spec_fold_blockmask,
}


def resolve_nucleation_mode(mode, use_simple_nucleation):
    if mode is None:
        mode = "legacy_simple" if bool(use_simple_nucleation) else "legacy_prob"
    if mode not in _NUCLEATION_KERNEL_RUNNERS:
        raise ValueError(f"Unknown nucleation_mode '{mode}'")
    return mode


def get_nucleation_kernel_runner(mode, no_outward=False):
    if no_outward:
        return _NUCLEATION_KERNEL_RUNNERS_SPEC[mode]
    return _NUCLEATION_KERNEL_RUNNERS[mode]


def get_nucleation_kernel_runner_fold(mode, no_outward=False):
    if no_outward:
        return _NUCLEATION_KERNEL_RUNNERS_SPEC_FOLD[mode]
    return _NUCLEATION_KERNEL_RUNNERS_FOLD[mode]


def precip_step_subblock_worker(task):
    """
    One worker for z-subblock nucleation: owns active cells with k in [k_lo, k_hi].
    Seed slab: k in [max(0, k_lo-1), min(n_z-1, k_hi+1)]. For each cell in the seed slab
    with oxidant > 0 and not full, perform up to oxidant[i,j,k] nucleation attempts,
    respecting oxidation_number (product cap) and only using active neighbours in [k_lo, k_hi].
    Inward/outward buffers are diffusion segments [count | dirs]; we decrement count and zero
    the freed dir slot so the segment stays consistent.
    task:
      - legacy: (cur_case_mp, k_lo, k_hi, plane_indexes, max_per_cell_oxidant, max_per_cell_active, ind_form)
      - with explicit z list: (..., ind_form, seed_slab_k_prepared)
      - block mask (heterogeneous blocks): (..., ind_form, block_mask_bits, cx, cy, cz)
      - block mask + explicit z list: (..., ind_form, block_mask_bits, cx, cy, cz, seed_slab_k_prepared)
    plane_indexes = x-axis (i) indices; worker's z range is [k_lo, k_hi].
    If seed_slab_k_prepared is provided, it is used directly (after bounds clamp).
    """
    block_mask_bits = None
    block_cells_x = 0
    block_cells_y = 0
    block_cells_z = 0
    if len(task) == 12:
        (
            cur_case_mp,
            k_lo,
            k_hi,
            plane_indexes,
            max_per_cell_o,
            max_per_cell_a,
            ind_form,
            block_mask_bits,
            block_cells_x,
            block_cells_y,
            block_cells_z,
            seed_slab_k_prepared,
        ) = task
    elif len(task) == 11:
        (
            cur_case_mp,
            k_lo,
            k_hi,
            plane_indexes,
            max_per_cell_o,
            max_per_cell_a,
            ind_form,
            block_mask_bits,
            block_cells_x,
            block_cells_y,
            block_cells_z,
        ) = task
        seed_slab_k_prepared = None
    elif len(task) == 8:
        cur_case_mp, k_lo, k_hi, plane_indexes, max_per_cell_o, max_per_cell_a, ind_form, seed_slab_k_prepared = task
    else:
        cur_case_mp, k_lo, k_hi, plane_indexes, max_per_cell_o, max_per_cell_a, ind_form = task
        seed_slab_k_prepared = None
    plane_indexes = np.asarray(plane_indexes, dtype=np.intp).ravel()
    shm_o = shared_memory.SharedMemory(name=cur_case_mp.oxidant_c3d_shm_mdata.name)
    n_i, n_j, n_z = cur_case_mp.oxidant_c3d_shm_mdata.shape
    k_lo = max(0, min(int(k_lo), int(n_z) - 1))
    k_hi = max(0, min(int(k_hi), int(n_z) - 1))
    if k_lo > k_hi:
        k_lo, k_hi = k_hi, k_lo
    n3 = n_i * n_j * n_z
    count_bytes_o = n3 * np.dtype(np.uint16).itemsize
    # Diffusion segment: [count (n³ uint16)][dirs (n³ × max_per_cell uint8)]; count view F-order
    oxidant = np.ndarray(
        (n_i, n_j, n_z),
        dtype=np.uint16,
        buffer=shm_o.buf,
        offset=0,
        order="F",
    )
    oxidant_dirs = np.ndarray(
        (n3, max_per_cell_o),
        dtype=np.uint8,
        buffer=shm_o.buf,
        offset=count_bytes_o,
    )
    shm_a = None
    if getattr(cur_case_mp, "active_c3d_shm_mdata", None) is not None and int(max_per_cell_a) > 0:
        shm_a = shared_memory.SharedMemory(name=cur_case_mp.active_c3d_shm_mdata.name)
        count_bytes_a = n3 * np.dtype(np.uint16).itemsize
        active = np.ndarray(
            (n_i, n_j, n_z),
            dtype=np.uint16,
            buffer=shm_a.buf,
            offset=0,
            order="F",
        )
        active_dirs = np.ndarray(
            (n3, max_per_cell_a),
            dtype=np.uint8,
            buffer=shm_a.buf,
            offset=count_bytes_a,
        )
    else:
        active = np.zeros((1, 1, 1), dtype=np.uint16)
        active_dirs = np.zeros((1, 1), dtype=np.uint8)
    
    shm_product_init = shared_memory.SharedMemory(name=cur_case_mp.precip_3d_init_shm_mdata.name)
    product_init = np.ndarray(cur_case_mp.precip_3d_init_shm_mdata.shape, dtype=cur_case_mp.precip_3d_init_shm_mdata.dtype, buffer=shm_product_init.buf)

    shm_state = shared_memory.SharedMemory(name=cur_case_mp.product_state_shm_mdata.name)
    product_state = np.ndarray(
        cur_case_mp.product_state_shm_mdata.shape,
        dtype=cur_case_mp.product_state_shm_mdata.dtype,
        buffer=shm_state.buf,
    )

    if seed_slab_k_prepared is None:
        k_seed_lo = max(0, k_lo)
        k_seed_hi = min(n_z - 1, k_hi + 1)
        # seed_slab_k = z range for this worker (plane_indexes are x indices, not z)
        seed_slab_k = np.arange(k_seed_lo, k_seed_hi, dtype=np.intp)
    else:
        seed_slab_k = np.asarray(seed_slab_k_prepared, dtype=np.intp).ravel()
        seed_slab_k = np.clip(seed_slab_k, 0, n_z - 1).astype(np.intp, copy=False)

    ox_num = cur_case_mp.oxidation_number
    nucl_prob = cur_case_mp.nucleation_probabilities
    n_cells = n_i

    active_check_offsets = np.asarray(ind_form[:9], dtype=np.int8)
    face_offsets = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int8,
    )
    if ox_num > 1:
        flat_neigh_offsets = np.vstack([face_offsets, [[0, 0, 0]]]).astype(np.int8)
    else:
        flat_neigh_offsets = face_offsets

    values_pp = np.asarray(nucl_prob.nucl_prob.values_pp, dtype=np.float64)
    const_a_pp = np.asarray(nucl_prob.const_a_pp, dtype=np.float64)
    const_b_pp = np.asarray(nucl_prob.const_b_pp, dtype=np.float64)
    const_c_pp = np.asarray(nucl_prob.const_c_pp, dtype=np.float64)
    const_d_pp = np.asarray(nucl_prob.const_d_pp, dtype=np.float64)
    seed = np.random.randint(0, 2**31)

    threshold_inward = int(getattr(cur_case_mp, "threshold_inward", 1))
    threshold_outward = int(getattr(cur_case_mp, "threshold_outward", 1))
    phase_id = int(getattr(cur_case_mp, "product_phase_id", 0))
    kernel_runner = getattr(cur_case_mp, "nucleation_kernel_runner", None)
    # If block-mask mode is active, always resolve a compatible runner (the cached
    # cur_case_mp.nucleation_kernel_runner is the legacy signature).
    if block_mask_bits is not None:
        kernel_runner = None
    if kernel_runner is None:
        mode = resolve_nucleation_mode(
            getattr(cur_case_mp, "nucleation_mode", None),
            getattr(cur_case_mp, "use_simple_nucleation", False),
        )
        no_outward = bool(getattr(cur_case_mp, "no_outward_nucleation", False))
        use_fold = bool(getattr(getattr(cur_case_mp, "nucleation_apply_fold", None), "value", False))
        # Fallback to Config if cur_case_mp doesn't carry the switch
        if not use_fold:
            try:
                from configuration import Config
                use_fold = bool(getattr(Config, "NUCLEATION_APPLY_FOLD", False))
            except Exception:
                use_fold = False
        if block_mask_bits is not None:
            if use_fold:
                kernel_runner = (_NUCLEATION_KERNEL_RUNNERS_SPEC_FOLD_BLOCKMASK if no_outward else _NUCLEATION_KERNEL_RUNNERS_FOLD_BLOCKMASK)[mode]
            else:
                kernel_runner = (_NUCLEATION_KERNEL_RUNNERS_SPEC_BLOCKMASK if no_outward else _NUCLEATION_KERNEL_RUNNERS_BLOCKMASK)[mode]
        else:
            if use_fold:
                kernel_runner = get_nucleation_kernel_runner_fold(mode, no_outward=no_outward)
            else:
                kernel_runner = get_nucleation_kernel_runner(mode, no_outward=no_outward)

    if block_mask_bits is None:
        kernel_runner(
            oxidant,
            oxidant_dirs,
            active,
            active_dirs,
            product_init,
            product_init,
            product_init,
            product_init,
            ox_num,
            threshold_inward,
            threshold_outward,
            seed_slab_k,
            plane_indexes,
            active_check_offsets,
            flat_neigh_offsets,
            values_pp,
            const_a_pp,
            const_b_pp,
            const_c_pp,
            const_d_pp,
            n_cells,
            seed,
            product_state,
            phase_id,
        )
    else:
        block_mask_bits = np.asarray(block_mask_bits, dtype=np.uint16)
        kernel_runner(
            oxidant,
            oxidant_dirs,
            active,
            active_dirs,
            product_init,
            product_init,
            product_init,
            product_init,
            ox_num,
            threshold_inward,
            threshold_outward,
            seed_slab_k,
            plane_indexes,
            active_check_offsets,
            flat_neigh_offsets,
            values_pp,
            const_a_pp,
            const_b_pp,
            const_c_pp,
            const_d_pp,
            n_cells,
            seed,
            product_state,
            phase_id,
            block_mask_bits,
            int(block_cells_x),
            int(block_cells_y),
            int(block_cells_z),
        )

    shm_o.close()
    if shm_a is not None:
        shm_a.close()
    shm_product_init.close()
    shm_state.close()
