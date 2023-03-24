# Efficient DNN Implementation on FPGA

The objective of this project is to optimize a pretrained deep neural network model by quantization/pruning/compression and implement it on a Xilinx PYNQ-Z2 board using the FINN Framework.

## Requirements

* Python >= 3.7 .
* [Pytorch](https://pytorch.org) >= 1.5.1 .
* Windows, Linux or macOS.
* GPU training-time acceleration (*Optional* but recommended).
* PYNQ-Z2 board with pynq v2.6 image (other versions may work but are not tested).

## Training DNN

We want to  train our own quantized neural network (QNN) using the Brevitas library, which supports quantization-aware training (QAT). This will involve experimenting with various quantization levels to find out the most energy-efficient model. Once the QNN model has been trained, it must be exported to FINN-ONNX format, which is a format compatible with the FINN Framework.

### Prerequisites
Brevitas library can be installed by running the following command:

```bash
pip install brevitas
```

Check out available info at https://xilinx.github.io/brevitas/getting_started for getting started with Brevitas.

Check out brevitas/notebooks for examples on how to use Brevitas.

## Exporting to FINN-ONNX

Next step is to export the trained QNN model to FINN-ONNX format. This can be done by following this [tutorial](https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import.ipynb).

## Creating the dataflow graph

FINN’s build_dataflow system must be applied to the exported model to create a dataflow graph for the FPGA platform. This step involves mapping each layer of the QNN to a Vivado HLS description, parallelizing each layer’s implementation to the appropriate degree, and utilizing on-chip FIFOs to connect the layers and create the full accelerator.
To create the dataflow graph, you can follow this [tutorial](https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb).

## Implementing the DNN on the FPGA

Implement the generated bitstream on the FPGA.
