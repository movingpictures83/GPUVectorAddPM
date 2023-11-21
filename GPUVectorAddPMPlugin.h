#ifndef GPUVECTORADDPMPLUGIN_H
#define GPUVECTORADDPMPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUVectorAddPMPlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

#endif
