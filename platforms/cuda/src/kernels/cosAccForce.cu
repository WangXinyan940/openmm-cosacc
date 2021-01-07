extern "C" __global__ 
void addForces(const FORCES_TYPE* __restrict__ forces, long long* __restrict__ forceBuffers, int* __restrict__ atomIndex, int numAtoms, int paddedNumAtoms) {
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        int index = atomIndex[atom];
        FORCES_TYPE addfrc = A * COS(2.0*PI*posq[index*4+2]/cell[8]) * masses[index];
        forceBuffers[atom] += (long long) (addfrc*0x100000000);
    }
}
