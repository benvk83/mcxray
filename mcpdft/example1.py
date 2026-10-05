from pyscf import gto, mp, mcpdft, mcscf

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
myhf = mol.RHF().run()
# Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
mymp = mp.UMP2(myhf).run()

noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
ncas, nelecas = (6,8)
otfnal = 'tPBE'
mycas = mcpdft.CASCI(myhf, otfnal, ncas, nelecas)
mycas.kernel(natorbs)
