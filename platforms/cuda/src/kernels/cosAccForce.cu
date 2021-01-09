extern "C" __global__ 
void addForces(const real*     __restrict__   massvec, 
               const real4*    __restrict__   posq, 
               long long*      __restrict__   forceBuffers, 
               int*            __restrict__   atomIndex, 
               FORCES_TYPE                    A, 
               int                            numAtoms, 
               int                            paddedNumAtoms) {
   for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
       int index = atomIndex[atom];
       FORCES_TYPE addfrc = A * COS(posq[index].z*TWOPIOVERLZ) * massvec[index];
       printf("Frc: %f\n", addfrc);
       forceBuffers[atom] += (long long) (addfrc*0x100000000);
   }
}

