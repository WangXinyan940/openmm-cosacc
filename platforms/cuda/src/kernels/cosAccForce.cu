extern "C" __global__ 
void addVels(const real4*    __restrict__   posq, 
             real4*          __restrict__   velm, 
             int*            __restrict__   atomIndex, 
             double                         A, 
             int                            numAtoms, 
             int                            paddedNumAtoms) {
   for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
       int index = atomIndex[atom];
       double addvel = A * COS(posq[index].z*TWOPIOVERLZ);
       velm[atom].x += addvel;
   }
}

