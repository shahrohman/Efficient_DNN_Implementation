#!/bin/bash
source /tools/Xilinx/Vivado/2022.1/settings64.sh
source /tools/Xilinx/Vitis_HLS/2022.1/settings64.sh

export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2022.1
export VIVADO_PATH=/tools/Xilinx/Vivado/2022.1
export VITIS_PATH=/tools/Xilinx/Vitis_HLS/2022.1
export FINN_DOCKER_TAG=maltanar/finn:dev_latest
export FINN_DOCKER_PREBUILT=1