#include "GPUVectorAddPMPlugin.h"

#define PINNED 1

void GPUVectorAddPMPlugin::input(std::string infile) {
   readParameterFile(infile);
}

void GPUVectorAddPMPlugin::run() {}

void GPUVectorAddPMPlugin::output(std::string outfile) {

  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

 inputLength = atoi(myParameters["N"].c_str());
 hostInput1 = (float*) malloc(inputLength*sizeof(float));
hostInput2 = (float*) malloc(inputLength*sizeof(float));
 hostOutput = (float*) malloc(inputLength*sizeof(float));
 std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["vector1"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < inputLength; ++i) {
        float k;
        myinput >> k;
        hostInput1[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+myParameters["vector2"]).c_str(), std::ios::in);
 for (i = 0; i < inputLength; ++i) {
        float k;
        myinput2 >> k;
        hostInput2[i] = k;
 }



#ifdef PINNED
  //@@ Part B: Allocate GPU memory here using pinned memory here
  cudaMallocHost((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMallocHost((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMallocHost((void **)&deviceOutput, inputLength * sizeof(float));
#endif

#ifndef PINNED
  //@@ Part A: Allocate GPU memory here using cudaMalloc here - this is
  //@@ non pinned version.
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));
#endif 


#ifdef PINNED
  //@@ Part B: GPUTK artificat to make the lab compatible for pinned memory
  memcpy(deviceInput1, hostInput1, inputLength * sizeof(float));
  memcpy(deviceInput2, hostInput2, inputLength * sizeof(float));
#endif

#ifndef PINNED
  //@@ Part A: Setup streams for non pinned version. Here in this example,
  //@@ we have 32 streams.
  unsigned int numStreams = 32; 
  cudaStream_t stream[numStreams];
  for(unsigned int i=0; i < numStreams; i++)
          cudaStreamCreate(&stream[i]);

  //@@ Part A: Create segments
  unsigned int numSegs = numStreams; 
  unsigned int segSize = (inputLength + numSegs -1 )/numSegs; 
 
  //@@ Part A: perform parallel vector addition with different streams. 
  for (unsigned int s =0; s<numSegs; s++){
          unsigned int start = s*segSize; 
          unsigned int end   = (start + segSize < (unsigned int) inputLength)? \
          start+segSize : inputLength;
          unsigned int Nseg  = end - start; 
          //@@ Part A: Copy data to the device memory in segments asynchronously
          cudaMemcpyAsync(&deviceInput1[start], &hostInput1[start], \
          Nseg*sizeof(float), cudaMemcpyHostToDevice, stream[s]);
          cudaMemcpyAsync(&deviceInput2[start], &hostInput2[start], \
          Nseg*sizeof(float), cudaMemcpyHostToDevice, stream[s]);
          const unsigned int numThreads = 32;
          const unsigned int numBlocks = (Nseg+numThreads-1)/numThreads; 

          //@@ Part A: Invoke CUDA Kernel
          vecAdd<<<numBlocks, numThreads, 0, stream[s]>>>(&deviceInput1[start], \
          &deviceInput2[start], &deviceOutput[start], Nseg);

          cudaMemcpyAsync(&hostOutput[start], &deviceOutput[start], \
          Nseg*sizeof(float), cudaMemcpyDeviceToHost, stream[s]);
  }
  //@@ Part A: Synchronize
  cudaDeviceSynchronize();
#endif

#ifdef PINNED
  //@@ Part B: Initialize the grid and block dimensions here
  dim3 blockDim(32);
  dim3 gridDim(ceil(((float)inputLength) / ((float)blockDim.x)));

  //@@ Part B: Launch the GPU Kernel here
  vecAdd<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput,
                                inputLength);
  cudaDeviceSynchronize();

  //@@ Part B: GPUTK artificat to make the lab compatible
  memcpy(hostOutput, deviceOutput, inputLength * sizeof(float));

#endif

        std::ofstream outsfile(outfile.c_str(), std::ios::out);
        for (i = 0; i < inputLength; ++i){
                outsfile << hostOutput[i];//std::setprecision(0) << a[i*N+j];
                outsfile << "\n";
        }


#ifndef PINNED 
  //@@ Destory cudaStream
  for(unsigned int i=0; i < numStreams; i++)
          cudaStreamDestroy(stream[0]);

  //@@ Part A: Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
#endif 
#ifdef PINNED
  //@@ Part B: Free the GPU memory here
  cudaFreeHost(deviceInput1);
  cudaFreeHost(deviceInput2);
  cudaFreeHost(deviceOutput);
#endif 

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

}

PluginProxy<GPUVectorAddPMPlugin> GPUVectorAddPMPluginProxy = PluginProxy<GPUVectorAddPMPlugin>("GPUVectorAddPM", PluginManager::getInstance());
