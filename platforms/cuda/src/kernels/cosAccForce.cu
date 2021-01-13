extern "C" __global__ 
void addForces(const real*     __restrict__   massvec, 
               const real4*    __restrict__   posq, 
               long long*      __restrict__   forceBuffers, 
               double                         A, 
               int                            numAtoms, 
               int                            paddedNumAtoms) {
   for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
       real addfrc = A * COS(posq[atom].z*TWOPIOVERLZ) * massvec[atom];
       forceBuffers[atom] += (long long) (addfrc*0x100000000);
   }
}

