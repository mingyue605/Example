
import ase
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import molecule, add_adsorbate, surface
from ase.io import write
from ase.constraints import FixAtoms
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms
from ase.optimize import BFGS
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.data.oc.utils.vasp import write_vasp_input_files
from ase.calculators.vasp import Vasp
from ase import Atoms, io
from vasp_flags import VASP_FLAGS # Download vasp_flags from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/data/oc/utils/vasp_flags.py
from ase.visualize import view
from fairchem.data.oc.utils.vasp import write_vasp_input_files, VASP_FLAGS
from ase.io.trajectory import Trajectory
from ase.io import extxyz

# Add an environment variable which contains instructions on how to execute VASP (https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html)
# This variable tells ASE how to execute VASP, using mpirun or any other MPI executor.
#%%bash
#export ASE_VASP_COMMAND="mpirun vasp_std"
calc = Vasp(command='mpiexec vasp_std')
# This specifies the location of pseudopotential directories required by VASP
os.environ['VASP_PP_PATH'] = './' #'/Users/mingyue/Documents/VASP/'

#export VASP_PP_PATH='./'  #"/Users/mingyue/Documents/VASP/"
def load_structure_from_materials_project(mp_id, api_key):
    """
    Load a structure from the Materials Project using the mp_id.

    Parameters:
    - mp_id: The Materials Project ID of the structure (e.g., 'mp-2657').
    - api_key: Your Materials Project API key.

    Returns:
    - ASE Atoms object representing the structure.
    """
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
    return AseAtomsAdaptor().get_atoms(structure)


def build_surface_slab(atoms, indices=(1, 1, 1), layers=5, vacuum=10.0, repeat_cell=(1, 1, 1)):
    """
    Build and expand a surface slab with specified Miller indices, number of layers, and vacuum thickness.

    Parameters:
    - atoms: The bulk structure as an ASE Atoms object.
    - indices: Miller indices for the surface (default is (1, 1, 1)).
    - layers: Number of atomic layers in the slab (default is 5).
    - vacuum: Thickness of the vacuum layer above the slab (default is 10.0 Å).
    - repeat_cell: Tuple specifying how many times to repeat the slab in (x, y, z) directions (default is (1, 1, 1)).

    Returns:
    - ASE Atoms object representing the expanded surface slab.
    """
    slab = surface(atoms, indices, layers)
    slab.center(vacuum=vacuum, axis=2)
    slab *= repeat_cell
    return slab


def add_adsorbate_to_slab(slab, adsorbate_name='CO2', height=2.0, rotation_axis='x', rotation_angle=30.0):
    """
    Add an adsorbate molecule to the slab at a specified height and rotation.

    Parameters:
    - slab: The slab to which the adsorbate will be added.
    - adsorbate_name: Name of the molecule to be added as an adsorbate (default is 'CO2').
    - height: Distance between the slab surface and the adsorbate (default is 2.0 Å).
    - rotation_axis: Axis of rotation ('x', 'y', or 'z') for the adsorbate (default is 'x').
    - rotation_angle: Angle in degrees to rotate the adsorbate (default is 30.0°).

    Returns:
    - ASE Atoms object representing the slab with the adsorbate added.
    """
    adsorbate = molecule(adsorbate_name)
    if rotation_angle != 0.0:
        adsorbate.rotate(a=rotation_angle, v=rotation_axis, center='COM')

    offset = (0.7, 0.4)  # Offset position of the adsorbate on the slab
    add_adsorbate(slab, adsorbate, height=height, offset=offset)

    return slab


def apply_tags_and_constraints(adslab, num_per_layer=12, num_adsorbate=3, layers_fix=3, fix_slab=False):
    """
    Apply tags to atoms in the slab and optionally apply constraints to fix atoms beneath the surface.

    Parameters:
    - adslab: ASE Atoms object representing the slab and adsorbate system.
    - num_per_layer: Number of atoms per layer in the slab (default is 12).
    - num_adsorbate: Number of adsorbate atoms (default is 3).
    - layers_fix: Number of layers of the slab to fix (default is 3).
    - fix_slab: Whether or not to apply constraints to fix slab atoms (default is False).

    Returns:
    - ASE Atoms object with tags and optional constraints applied.
    """
    num_atoms = len(adslab)
    tags = np.zeros(num_atoms, dtype=int)

    # Tag atoms in the fixed layers beneath the surface
    tags[:num_per_layer * layers_fix] = 0

    # Tag surface atoms
    if num_adsorbate > 0:
        tags[num_per_layer * layers_fix:num_atoms - num_adsorbate] = 1
        # Tag adsorbate atoms
        tags[-num_adsorbate:] = 2
    else:
        tags[num_per_layer * layers_fix:] = 1

    adslab.set_tags(tags)

    # Apply constraints if needed
    if fix_slab:
        cons = FixAtoms(indices=[atom.index for atom in adslab if atom.tag == 0])
        adslab.set_constraint(cons)

    adslab.center(vacuum=13.0, axis=2)
    adslab.set_pbc(True)

    return adslab


def write_output_file(atoms, output_dir, filename='output.xyz'):
    """
    Write the structure to an output file compatible with visualization tools like Ovito.

    Parameters:
    - atoms: The ASE Atoms object to write.
    - output_dir: Directory where the output file will be saved.
    - filename: The name of the output file (default is 'output.xyz').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)
    write(filepath, atoms)
    print(f"Structure written to {filepath} for visualization in Ovito.")


def calculate_adsorption_energy(slab_energy, slab_with_adsorbate_energy, adsorbate_atoms):
    """
    Calculate the adsorption energy for the slab + adsorbate system.

    Parameters:
    - slab_energy: Energy of the clean slab (without adsorbate).
    - slab_with_adsorbate_energy: Energy of the slab with adsorbate.
    - adsorbate_atoms: ASE Atoms object representing the adsorbate.

    Returns:
    - Adsorption energy in eV.
    """
    gas_reference_energies = {'H': -3.477, 'N': -8.083, 'O': -7.204, 'C': -7.282}
    adsorbate_reference_energy = sum(gas_reference_energies[atom] for atom in adsorbate_atoms.get_chemical_symbols())

    adsorption_energy = slab_with_adsorbate_energy - slab_energy - adsorbate_reference_energy

    print(f"Adsorbate reference energy = {adsorbate_reference_energy:.2f} eV")
    print(f"Adsorption energy: {adsorption_energy:.2f} eV")

    return adsorption_energy


if __name__ == '__main__':
    api_key = 'q7kwIAVYtobJWJuWfxmin8UJbeGVMecs'
    mp_ids = ['mp-656887','mp-19009']  # List of mp_ids you want to loop through 'mp-18759', 'mp-1232659', 'mp-644514'

    for mp_id in mp_ids:
        print(f"Processing {mp_id}...")

        # Create directories for the current material
        output_dir = f'Ni/{mp_id}'
        os.makedirs(f'{output_dir}/traj', exist_ok=True)
        os.makedirs(f'{output_dir}/vasp', exist_ok=True)

        # Step 1: Load structure from the Materials Project
        atoms = load_structure_from_materials_project(mp_id, api_key)

        # Step 2: Build the surface slab
        slab = build_surface_slab(atoms, indices=(1, 1, 1), layers=5, vacuum=10.0, repeat_cell=(1, 1, 1))
        slab_with_tags = apply_tags_and_constraints(slab, num_per_layer=12, num_adsorbate=0, layers_fix=0, fix_slab=False)

        SLABBB = slab_with_tags.copy()
        write_vasp_input_files(slab_with_tags, outdir=f'{output_dir}/vasp/{mp_id}_slab')

        # Step 3: Add CO₂ adsorbate to the slab
        slab_with_adsorbate = add_adsorbate_to_slab(slab, adsorbate_name='CO2', height=3, rotation_axis='x', rotation_angle=60.0)

        # Step 4: Apply tags and constraints to the slab + adsorbate system
        adslab_with_tags_and_constraints = apply_tags_and_constraints(slab_with_adsorbate, num_per_layer=12, num_adsorbate=3, layers_fix=0, fix_slab=False)
        write_vasp_input_files(adslab_with_tags_and_constraints, outdir=f'{output_dir}/vasp/{mp_id}_adslab')

        # Step 5: Set up the OCPCalculator and relax the slab + adsorbate system
        checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/fairchem_checkpoints/')
        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)

        slab_with_tags.set_calculator(calc)
        SLABBB.set_calculator(calc)
        adslab_with_tags_and_constraints.set_calculator(calc)

        slab_traj_dir = f'{output_dir}/traj/{mp_id}_relax_slab.traj'

        dyn = BFGS(SLABBB, trajectory=slab_traj_dir)
        dyn.run(fmax=0.05, steps=500)

        traj_dir = f'{output_dir}/traj/{mp_id}_relax_abs.traj'
        dyn = BFGS(adslab_with_tags_and_constraints, trajectory=traj_dir)
        dyn.run(fmax=0.05, steps=500)

        # Step 6: Write the relaxed slab and relaxed slab + adsorbate to VASP files
        slab_traj = ase.io.read(slab_traj_dir, ":")
        adslab_traj = ase.io.read(traj_dir, ":")
        write_vasp_input_files(slab_traj[-1], outdir=f'{output_dir}/vasp/{mp_id}_slab_relaxed')
        write_vasp_input_files(adslab_traj[-1], outdir=f'{output_dir}/vasp/{mp_id}_adslab_relaxed')

        # Step 7: Calculate adsorption energy
        adsorbate_atoms = Atoms('CO2')

        relaxed_slab = ase.io.read(slab_traj_dir)
        slab_energy = relaxed_slab.get_potential_energy()
        print(f"Slab energy = {slab_energy:.2f} eV")

        relaxed_adslab = ase.io.read(traj_dir)
        slab_with_adsorbate_energy = relaxed_adslab.get_potential_energy()
        print(f"Adsorbate+slab energy = {slab_with_adsorbate_energy:.2f} eV")

        adsorption_energy = calculate_adsorption_energy(slab_energy, slab_with_adsorbate_energy, adsorbate_atoms)
        print(f"Adsorption energy for {mp_id}: {adsorption_energy:.2f} eV")

