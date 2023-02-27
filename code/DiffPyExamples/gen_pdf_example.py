import torch # Pytorch package
from diffpy.structure import Structure, Atom # Diffpy-CMI package
from diffpy.srreal.pdfcalculator import DebyePDFCalculator # Diffpy-CMI package
from mendeleev import element # Mendeelev package (pip install mendeleev)

""" Example pointcloud with atomic numbers """
xyz_coordinates = torch.rand((5,3)) * 10 # 5 Atoms, 3 coordinates (x,y,z)
atom_species = torch.tensor([79, 8, 8, 8, 8], dtype=torch.uint8) # 1 Gold atom and 4 Oxygen atoms

""" Creating a Structure object (we use a the mendeleev package to convert the integers to element symbol strings) """
structure = Structure()
for atom,xyz in zip(atom_species, xyz_coordinates):
    structure.append(Atom(element(atom.item()).symbol,xyz))

""" Setting vibration modes (perform this step every time you create a new structure) """
structure.B11 = 0.3 # Keep to 0.3, isotropic vibration on first axis
structure.B22 = 0.3 # Keep to 0.3, isotropic vibration on second axis
structure.B33 = 0.3 # Keep to 0.3, isotropic vibration on third axis
structure.B12 = 0 # Keep at 0, anisotropic vibration
structure.B13 = 0 # Keep at 0, anisotropic vibration
structure.B23 = 0 # Keep at 0, anisotropic vibration

""" Setting up the simulation parameters (set these only once) """
pdf_params = dict(
    rmin = 0, # minimum value at which the pdf is evaluated
    rmax = 30, # maximum value at which the pdf is evaluated
    rstep = 0.01, # step-size of evaluation of pdf

    qmin = 0.8, # dont worry about this, keep at this value
    qmax = 30, # dont worry about this, keep at this value
    delta2 = 0, # dont worry about this, keep at this value
    qdamp = 0.01, # dont worry about this, keep at this value
)

""" Setup PDF-Calculator object (initialise only once) """
pdf_calculator = DebyePDFCalculator(**pdf_params)

""" Calculate PDF of structure by passing the structure to the calculator, you get both x-axis "r" and y-axis "G(r)", the PDF. """
r, gr = pdf_calculator(structure) # Will produce a numpy array
r = torch.tensor(r, dtype=torch.float32)
gr = torch.tensor(gr, dtype=torch.float32)

# If you want to normalise
gr /= (torch.max(gr) + 1e-12)
