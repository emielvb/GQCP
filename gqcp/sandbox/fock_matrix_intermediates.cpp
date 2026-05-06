
#include "Basis/SpinorBasis/GSpinorBasis.hpp"
#include "Basis/Transformations/GTransformation.hpp"
#include "QCModel/HF/GHF.hpp"

#include <Eigen/Dense>

#include <iostream>

int main() {

    using Scalar = GQCP::complex;

    // --- molecule ---
    const GQCP::Nucleus h1 {1, 0.0, 0.0, 0.0};
    const GQCP::Nucleus h2 {1, 1.0, 0.0, 0.0};
    const GQCP::Nucleus h3 {1, 0.0, 1.0, 0.0};
    const GQCP::Molecule molecule {{h1, h2, h3}};
    const auto N = molecule.numberOfElectrons();

    // --- magnetic field ---
    const auto B = GQCP::HomogeneousMagneticField {{0.0, 0.0, -0.2}};

    // --- spinor basis ---
    auto spinor_basis =
        GQCP::GSpinorBasis<Scalar, GQCP::LondonGTOShell> {molecule, "STO-3G", B};

    // --- Hamiltonian (second-quantized, scalar/AO basis) ---
    const auto sq_hamiltonian =
        spinor_basis.quantize(GQCP::FQMolecularPauliHamiltonian(molecule, B));

    const auto S = spinor_basis.overlap();                 // overlap operator

    // diagonalise Hcore to get initial guess.
    // taken from GHFSCFEnvironment.hpp

    const auto& H_core = sq_hamiltonian.core().parameters();

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> generalized_eigensolver {H_core, S.parameters()};
    const GQCP::GTransformation<Scalar> C_initial {generalized_eigensolver.eigenvectors()};

    // ------------------------------------------------------------------
    // 2. Build density matrix from Hcore guess
    // ------------------------------------------------------------------

    // P = C* D C^T, where D occupies the lowest N spinors
    const auto P =
        GQCP::QCModel::GHF<Scalar>::calculateScalarBasis1DM(C_initial, N);

    // ------------------------------------------------------------------
    // 3. Compute Coulomb (J) and Exchange (K)
    // ------------------------------------------------------------------

    const auto J =
        GQCP::QCModel::GHF<Scalar>::calculateScalarBasisDirectMatrix(P, sq_hamiltonian);

    const auto K =
        GQCP::QCModel::GHF<Scalar>::calculateScalarBasisExchangeMatrix(P, sq_hamiltonian);

    // ------------------------------------------------------------------
    // 4. Print some diagnostics
    // ------------------------------------------------------------------

    std::cout << "C matrix from Hcore diagonalisation:" << C_initial.matrix() << "\n";
    std::cout << "Number of spinors: " << P.numberOfOrbitals() << "\n";
    std::cout << "J = " << J.parameters() << "\n";
    std::cout << "K = " << K.parameters() << "\n";

    return 0;
}
